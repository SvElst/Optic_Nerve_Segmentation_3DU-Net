#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Apply a trained model to an independent test dataset.

"""
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
gpus = tf.config.experimental.list_physical_devices('GPU')


for gpu in gpus:
 	tf.config.experimental.set_memory_growth(gpu,True)
logical_gpus = tf.config.experimental.list_logical_devices('GPU')
print(len(gpus), "Physical GPUs,",len(logical_gpus), "Logical GPUs")

# Import self created modules
import load_data
from DataGenerator import DataGenerator
import predict
from losses_metrics import dice_coefficient, dice_coefficient_loss

# Import packages
from tensorflow.python.keras.models import load_model
import numpy as np
import os.path
import glob
import optparse

# Set random seed for reproducibility
np.random.seed(123)  # for reproducibility

#%% Set parser options
parser = optparse.OptionParser()
parser.add_option('--model', action="store", dest="model_fname", default="")
parser.add_option('--datadir', action="store", dest="datadir",help="Data directory",default="")
parser.add_option('--folderdir', action="store", dest="folderdir",help="Folder directory",default="") 
parser.add_option('--batch_size', action="store", type="int", dest="batch_size", help="Batch size",default=1)
parser.add_option('--channels',action="append",type="string",dest="channels", default=[])  # Important one last
parser.add_option('--threshold', action="store", type="float", dest="threshold", help="",default=0.5)        # Thresholds to binarize image
parser.add_option('--classes',action="append",type="string", dest="classes", default=[])
parser.add_option('--inputsize',action="store",type="int", dest="inputsize", nargs=3) 

options, args = parser.parse_args()

# Options
batch_size = int(options.batch_size)
patch_size = list(options.inputsize)
folderdir = options.folderdir
channels = options.channels
classes = options.classes
if len(classes)>1:      # In case of multiclass segmentation; include a background class
    classes = ['Background']+classes
nclasses=len(classes)   
#%% Prepare dataset
# Subjects
leftright = ['OD','OS']
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
fp_images_t1c = np.memmap(memmap_path + 'memmaps/images_t1c.dat',dtype=np.float32,mode='w+',shape=fp_shape)
fp_images_t2 = np.memmap(memmap_path + 'memmaps/images_t2.dat',dtype=np.float32,mode='w+',shape=fp_shape)
fp_images_gt =  np.memmap(memmap_path + 'memmaps/images_gt.dat',dtype=np.short,mode='w+',shape=fp_shape)

# Initialize to zero / false
del fp_images_fiesta, fp_images_t1c, fp_images_t2, fp_images_gt

#%% Load data
subjid_list = load_data.load(SUBJIDS,datadir,nclasses,classes, channels, patch_size, memmap_path, fp_shape)
       
#%% Retrieve data
# Open existing memmaps and save in dictionary 
d_memmap = {}
fp_images_gt =  np.memmap(memmap_path + 'memmaps/images_gt.dat',dtype=np.short,mode='r',shape=fp_shape)

for channel in channels:    # Open required scans 
    fp_images = np.memmap(memmap_path + 'memmaps/images_{}.dat'.format(channel),dtype=np.float32 ,mode='r',shape=fp_shape)
    d_memmap[channel] = fp_images

d_memmap['gt'] = fp_images_gt
#%% Define parameters for model testing
# Create data generators
params = {'batch_size': batch_size,
      'n_classes':      nclasses,
      'channels':       channels,
      'shuffle':        0,
      'mode':           'eval',
      'dim':            (fp_shape[1:4])}	

# Dataset indices
indices = list(range(len(subjid_list)))
test_generator = DataGenerator(indices, d_memmap, **params)

# Folders to save results
path = os.path.dirname(os.path.dirname(options.model_fname)) + '/'
result_path = path + 'test_results/'
if not os.path.exists(result_path):
     os.mkdir(result_path)

#%% Prediction
print ('Start testing now') 
# Load model
model = load_model(options.model_fname, custom_objects={'dice_coefficient_loss':dice_coefficient_loss, 'dice_coefficient':dice_coefficient})

#Predict on the test data
predict.predict(model, test_generator, indices, d_memmap, nclasses, subjid_list, path=path,result_path=result_path, threshold=options.threshold, save_images=True)