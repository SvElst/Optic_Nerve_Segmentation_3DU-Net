#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train model using k-fold crossvalidation or hold-out validation
"""
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
gpus = tf.config.experimental.list_physical_devices('GPU')


for gpu in gpus:
 	tf.config.experimental.set_memory_growth(gpu,True)
logical_gpus = tf.config.experimental.list_logical_devices('GPU')
print(len(gpus), "Physical GPUs,",len(logical_gpus), "Logical GPUs")

# Import self created modules
import unet_model
import load_data
from DataGenerator import DataGenerator
import predict
from losses_metrics import dice_coefficient, dice_coefficient_loss

# Import packages 
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras import backend as K
from tensorflow.python import keras
import numpy as np
import os.path
import glob
import optparse
from sklearn.model_selection import KFold

#Set random seed for reproducibility
np.random.seed(123)

#%% Set parser options
parser = optparse.OptionParser()

# Data/save options
parser.add_option('--datadir', action="store", dest="datadir",help="Data directory",default="")
parser.add_option('--folderdir', action="store", dest="folderdir",help="Folder directory",default="")
parser.add_option('--output_prefix', action="store", type="string", dest="output_prefix",default="Test")
# Input options
parser.add_option('--inputsize',action="store",type="int", dest="inputsize", nargs=3)
parser.add_option('--channels',action="append",type="string",dest="channels", default=[])                      # Options: fiesta, t2, t1c
parser.add_option('--classes',action="append",type="string", dest="classes", default=[])                       # Options: Opticus, Eye, Tumor
# Network options
parser.add_option('--net',action="store",type="string",dest="network", default='unet')                         # Options: unet, resnet
parser.add_option('--filtersize', action="store", type="int", dest="filtersize", help="",default=32)
parser.add_option('--depth', action="store", type="int", dest="depth", help="",default=5)
parser.add_option('--attGate',action="store",type="int", dest="AttGating", default=0)                          # 0 = False, 1 = True          
# Training/Validation approach
parser.add_option('--nfolds', action="store",type="int", dest="nfolds",help="number of folds",default=10)
parser.add_option('--valindices',action="store",type="int", dest="valindices", nargs=-1, default=[])           # Indices of validation subjects
# Learning options
parser.add_option('--lr',action="store",type="float", dest="lr", default=0.0001)  
parser.add_option('--epochs', action="store", type="int", dest="epochs", help="",default=120) 
parser.add_option('--batch_size', action="store", type="int", dest="batch_size", help="Batch size",default=1)         
# Augmentation options
parser.add_option('--augment_flip',action="store", type="float", dest="augment_flip", default=0.5)
parser.add_option('--augment_scale',action="store", type="float", dest="augment_scale", default=0.5)
parser.add_option('--augment_intensityShift',action="store", type="float", dest="augment_intensityShift", default=0.3)
parser.add_option('--augment_noise',action="store", type="float", dest="augment_noise", default=0.3)
parser.add_option('--augment_blur',action="store", type="float", dest="augment_blur", default=0.3)

options, args = parser.parse_args()
val_indices= list(options.valindices)           # Subjects in independent validation set

# Options
epochs = int(options.epochs)
batch_size = int(options.batch_size)
patch_size=list(options.inputsize)
folderdir = options.folderdir
channels = options.channels                 
classes = options.classes
if len(classes)>1:              # In case of multiclass segmentation; include a background class
    classes = ['Background']+classes
nclasses=len(classes)

#%% Prepare datset
# Subjects
datadir = options.datadir
SUBJIDS = [os.path.basename(x) for x in glob.glob(datadir+ "/*")]

# Create memory map folders
cuda_core = os.environ['CUDA_VISIBLE_DEVICES']
if not os.path.exists(folderdir +'/Memmaps'): 
    os.mkdir(folderdir +'/Memmaps')
memmap_path = folderdir +'/Memmaps/cuda_' + cuda_core + '_'
if not os.path.exists(memmap_path + 'memmaps'):
    os.mkdir(memmap_path + 'memmaps')

# Initialize memmory maps    
fp_shape = (len(SUBJIDS)*2,patch_size[0],patch_size[1], patch_size[2])
fp_images_fiesta = np.memmap(memmap_path + 'memmaps/images_fiesta.dat',dtype=np.float32,mode='w+',shape=fp_shape)
fp_images_gt =  np.memmap(memmap_path + 'memmaps/images_gt.dat',dtype=np.short,mode='w+',shape=fp_shape)

# Initialize to zero / false
del fp_images_fiesta,  fp_images_gt

#%% Load data
""" For all subjects in list subjids, data is loaded into the memmory maps. 
    Output is a dictionary with subjids and hemis and scan parameters"""
subjid_list = load_data.load(SUBJIDS,datadir,nclasses,classes, channels, patch_size, memmap_path, fp_shape)

#%% Retrieve data
""" Open existing memmaps and save in dictionary """
d_memmap = {}
fp_images_gt =  np.memmap(memmap_path + 'memmaps/images_gt.dat',dtype=np.short,mode='r',shape=fp_shape)

for channel in channels:    # Open required scans 
    fp_images = np.memmap(memmap_path + 'memmaps/images_{}.dat'.format(channel),dtype=np.float32 ,mode='r',shape=fp_shape)
    d_memmap[channel] = fp_images

d_memmap['gt'] = fp_images_gt
#%% Define parameters for model trainng
lr = options.lr    # Learning rate

# Create data generators
params = {'batch_size': batch_size,
      'n_classes':      nclasses,
      'channels':       channels,
      'shuffle':        1,
      'mode':           'train',
      'dim':            (fp_shape[1:4])}

params_val = {'batch_size': batch_size,
      'n_classes':      nclasses,
      'channels':       channels,
      'shuffle':        0,
      'mode':           'eval',
      'dim':            (fp_shape[1:4])}

augment_opts = {'Flip':options.augment_flip,
                'Scale':options.augment_scale,
                'Intensity_shift':options.augment_intensityShift,
                'Additive_noise': options.augment_noise,
                'Gaussian_blur':options.augment_blur,
                }
	  
# Create folders to save models
path = folderdir + '/Output'
if not os.path.exists(path):
    os.mkdir(path) 
if not os.path.exists(path + '/' + options.output_prefix):
    os.mkdir(path + '/' + options.output_prefix)       
if not os.path.exists(path + '/' + options.output_prefix +'/models'):
    os.mkdir(path + '/' + options.output_prefix + '/models')
if not os.path.exists(path + '/' + options.output_prefix + '/images'):
    os.mkdir(path + '/' + options.output_prefix + '/images')
if not os.path.exists(path + '/' + options.output_prefix + '/validation_results'):
    os.mkdir(path + '/' + options.output_prefix + '/validation_results')
    if not os.path.exists(path + '/' + options.output_prefix + '/history'):
        os.mkdir(path + '/' + options.output_prefix + '/history')

#%% Start training
if options.nfolds>1:  
    " K-fold crossvalidation: "
    kf = KFold(n_splits = int(options.nfolds), random_state=7, shuffle = True)
    split = kf.split(SUBJIDS)

    fold=1
    for train_subject, val_subject in split:

        # Split train and validation subjects 
        subs_train = [SUBJIDS[index] for index in train_subject]
        train_indices = [x for x in range(len(subjid_list)) if subjid_list[x]['subjid'] in subs_train]
        val_indices = [x for x in range(len(subjid_list)) if subjid_list[x]['subjid'] not in subs_train]
    
        # Define data generators
        training_generator = DataGenerator(train_indices, d_memmap, **params, augment_opts=augment_opts)
        validation_generator = DataGenerator(val_indices, d_memmap, **params_val)

        # Traing model
        opt = tf.keras.optimizers.Adam(lr)
        model = unet_model.build_unet((patch_size[0], patch_size[1], patch_size[2],len(channels)), n_classes=nclasses,filtersize=options.filtersize, depth=options.depth, net=options.network, AttGating=options.AttGating)
        model.compile(optimizer = opt, loss=dice_coefficient_loss, metrics=[dice_coefficient])
        check_path = path + '/' + options.output_prefix + f'/models/checkpoint_fold{fold}.h5' # _{epoch:02d}
        ModelCheckpoint = tf.keras.callbacks.ModelCheckpoint(filepath=check_path, 
                                                monitor = "val_loss", verbose = 1, save_best_only = True, 
                                                save_weights_only = True, mode = "min", 
                                                save_freq = "epoch");
        EarlyStop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=40)
        LR_scheduler=tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10)
        
        callbacklist = [ModelCheckpoint, EarlyStop]#, LR_scheduler]
        print('"""""')
        print (f'Start training now - Fold {fold}')
        print('"""""')
        history=model.fit(training_generator, epochs=epochs, validation_data=validation_generator, callbacks=callbacklist, verbose=1)
        historysummary = open(path + '/' + options.output_prefix + '/history/' + f"History_fold{fold}.txt","w+")
        historysummary.write('Dice: ' + str(history.history['dice_coefficient']) + '\n')
        historysummary.write('Validation Dice: ' + str(history.history['val_dice_coefficient']) + '\n')
        historysummary.close()

        # If checkpoint:
        if callbacklist:    
            model.load_weights(check_path)  # Only if checkpoints is on
        model.save(path + '/' + options.output_prefix + f'/models/model_fold{fold}.h5')
        
        # Predict on hold-out validation set
        model = load_model(path + '/' + options.output_prefix +f'/models/model_fold{fold}.h5', custom_objects={'dice_coefficient_loss':dice_coefficient_loss, 'dice_coefficient':dice_coefficient})
        pred_path = path + '/' + options.output_prefix +'/' 
        result_path = path + '/' + options.output_prefix + f'/validation_results/fold_{fold}/'
        if not os.path.exists(result_path):
            os.mkdir(result_path)

        print('"""""')
        print (f'Start validation now - Fold {fold}')
        print('"""""')
        predict.predict(model, validation_generator, val_indices, d_memmap, nclasses, subjid_list, path=pred_path,result_path=result_path, threshold=0.5, save_images=False)

        # Destroy TF graph and remove model
        K.clear_session()
        del model, opt, history
        
        fold+=1

else:                
    " Hold-out validation: "
    train_indices = list(range(len(subjid_list)))   # Subjects in training set
    val_indices= list(options.valindices)           # Subjects in independent validation set
    for val in val_indices:
        train_indices.remove(val)
    
    # Define data generators
    training_generator = DataGenerator(train_indices, d_memmap, **params, augment_opts=augment_opts)
    validation_generator = DataGenerator(val_indices, d_memmap, **params_val)

    # Traing model
    opt = tf.keras.optimizers.Adam(lr)
    model = unet_model.build_unet((patch_size[0], patch_size[1], patch_size[2],len(channels)), n_classes=nclasses,filtersize=options.filtersize, depth=options.depth, net=options.network, AttGating=options.AttGating)
    model.compile(optimizer = opt, loss=dice_coefficient_loss, metrics=[dice_coefficient])
    check_path = path + '/' + options.output_prefix + '/models/checkpoint.h5' # _{epoch:02d}
    ModelCheckpoint = tf.keras.callbacks.ModelCheckpoint(filepath=check_path, 
                                            monitor = "val_loss", verbose = 1, save_best_only = True, 
                                            save_weights_only = True, mode = "min", 
                                            save_freq = "epoch");

    callbacklist = [ModelCheckpoint]
    print('"""""')
    print ('Start training now - all data') 
    history=model.fit(training_generator, epochs=epochs, validation_data=validation_generator, callbacks=callbacklist, verbose=1)
    historysummary = open(path + '/' + options.output_prefix + '/history/' + "History.txt","w+")
    historysummary.write('Dice: ' + str(history.history['dice_coefficient']) + '\n')
    historysummary.write('Validation Dice: ' + str(history.history['val_dice_coefficient']) + '\n')
    historysummary.close()
    
    if callbacklist:
        model.load_weights(check_path)  # Only if checkpoints is on
    model.save(path + '/' + options.output_prefix + '/models/model.h5')

    print('"""""')
    print ('Start validation now ')
    # Predict on hold-out validation set
    pred_path = path + '/' + options.output_prefix +'/' 
    result_path = path + '/' + options.output_prefix + '/validation_results/'
    if not os.path.exists(result_path):
        os.mkdir(result_path)
    predict.predict(model, validation_generator, val_indices, d_memmap, nclasses, subjid_list, path=pred_path,result_path=result_path, threshold=0.5, save_images=True)


#%% Save a summary of parameters used in this model training
filesummary = open(path + '/' + options.output_prefix + "/Model_summary.txt","w+")
filesummary.write('Model name: ' + options.output_prefix + '\n')
filesummary.write('Model type: ' + options.network + '\n')
filesummary.write('Use Attention Gating: ' + str(bool(options.AttGating)) + '\n')
filesummary.write('Number of subjects: ' + str(len(SUBJIDS)) + '\n')
filesummary.write('Number of classes: ' + str(nclasses) + '\n')
filesummary.write('Classes: ' + str(classes) + '\n')
filesummary.write('Channels: ' + str(channels) + '\n') 
filesummary.write('Patch size: ' + str(patch_size) + '\n')
filesummary.write('Batch size: ' + str(batch_size) + '\n')
filesummary.write('Epochs: ' + str(epochs) + '\n')
filesummary.write('Folds: ' + str(options.nfolds) + '\n')
filesummary.write('Learnin rate: ' + str(lr) + '\n')
filesummary.write('Filtersize: ' + str(options.filtersize) + '\n')
filesummary.write('Callbacks: ' + str(callbacklist) + '\n')
filesummary.close()