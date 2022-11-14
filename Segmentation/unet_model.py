#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generalized U-Net for 2D and 3D applications.
Enabels multiclass or binary segmentation.
With options to use residual blocks or attention gating blocks.

"""
# Import packages
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input, Activation, Concatenate, Add, Multiply, Conv3D, Conv2D, MaxPooling3D, MaxPooling2D, Conv3DTranspose, Conv2DTranspose, UpSampling3D, UpSampling2D, BatchNormalization, Dropout, Lambda 

#%%
" Blocks "
def expend_as(tensor, rep):
    # https://github.com/robinvvinod/unet/blob/ce70fa9d4b0fd5c017e32278217ed84b1228fb56/layers3D.py#L179
    # Anonymous lambda function to expand the specified axis by a factor of argument, rep.
    # If tensor has shape (512,512,N), lambda will return a tensor of shape (512,512,N*rep), if specified axis=2
    return Lambda(lambda x, repnum: K.repeat_elements(x, repnum, axis=-1),
                       arguments={'repnum': rep})(tensor)
def conv_block(inputs, num_filters, input_shape):
    # Define 2D or 3D network
    if len(input_shape) == 3:
        Conv = Conv2D
    elif len(input_shape) == 4:
        Conv = Conv3D
        
    x = Conv(num_filters, 3, padding="same")(inputs)
    x = BatchNormalization()(x)   #Not in the original network. 
    x = Activation("relu")(x)

    x = Conv(num_filters, 3, padding="same")(x)
    x = BatchNormalization()(x)  #Not in the original network
    x = Activation("relu")(x)

    return x

def res_block(inputs,num_filters,input_shape, strides=1):
    # Define 2D or 3D network
    if len(input_shape) == 3:
        Conv = Conv2D
    elif len(input_shape) == 4:
        Conv = Conv3D

    x = Conv(num_filters, 3, padding="same")(inputs)
    x = BatchNormalization()(x)   #Not in the original network. 
    x = Activation("relu")(x)

    x = Conv(num_filters, 3, padding="same")(x)
    x = BatchNormalization()(x)  #Not in the original network

    shape_input = tf.shape(inputs)
    if shape_input[-1] != num_filters:
        shortcut = Conv(num_filters,(1, 1, 1),strides=strides,padding='same',use_bias=False)(inputs)
        shortcut = BatchNormalization()(shortcut)
    else:
        shortcut = inputs

    x = Add()([x,shortcut])
    x = Activation('relu')(x)
    return x

def att_block(x, g, filter_shape, input_shape):
    # https://github.com/robinvvinod/unet/blob/ce70fa9d4b0fd5c017e32278217ed84b1228fb56/layers3D.py#L189
    """
    self gated attention, attention mechanism on spatial dimension
    :param x: input feature map
    :param gating: gate signal, feature map from the lower layer
    :param filter_shape: number of channels
    :return: attention weighted on spatial dimension feature map
    """  
    if len(input_shape) == 3:
        Conv = Conv2D
        UpSampling = UpSampling2D
    elif len(input_shape) == 4:
        Conv = Conv3D
        UpSampling = UpSampling3D
    
    shape_x = K.int_shape(x)
    shape_g = K.int_shape(g)

    # Getting the gating signal to the same number of filters as the filter_shape
    phi_g = Conv(filters=filter_shape,
                   kernel_size=1,
                   strides=1,
                   padding='same')(g)

    # Getting the x signal to the same shape as the gating signal
    theta_x = Conv(filters=filter_shape,
                     kernel_size=2,
                     strides=2,
                     padding='same')(x)

    # Element-wise addition of the gating and x signals
    add_xg = Add()([phi_g, theta_x])
    act_xg = Activation('relu')(add_xg)

    # 1x1x1 convolution
    psi = Conv(filters=1, kernel_size=1, padding='same')(act_xg)
    psi = Activation('sigmoid')(psi)
    shape_sigmoid = K.int_shape(psi)

    # Upsampling psi back to the original dimensions of x signal
    if len(input_shape) == 3:
        upsample_psi = UpSampling(size=(shape_x[1] // shape_sigmoid[1], shape_x[2] // shape_sigmoid[2]))(psi)
    elif len(input_shape) == 4:
        upsample_psi =  UpSampling(size=(shape_x[1] // shape_sigmoid[1], shape_x[2] // shape_sigmoid[2],
                  shape_x[3] // shape_sigmoid[3]))(psi)

    # Expanding the filter axis to the number of filters in the original x signal
    upsample_psi = expend_as(upsample_psi, shape_x[-1])

    # Element-wise multiplication of attention coefficients back onto original x signal
    attn_coefficients = Multiply()([upsample_psi, x])

    # Final 1x1x1 convolution to consolidate attention signal to original x dimensions
    output = Conv(filters=shape_x[-1],
                    kernel_size=1,
                    strides=1,
                    padding='same')(attn_coefficients)
    output = attn_coefficients
    output = BatchNormalization()(output)
    return output

#%% Build U-Net
" Construct U-Net"
def build_unet(input_shape, n_classes, filtersize, depth, net='unet', AttGating=False):
    print('Network type: ', net)
    # Define 2D or 3D network
    if len(input_shape) == 3:
        Conv = Conv2D
        ConvTranspose = Conv2DTranspose
        MaxPooling = MaxPooling2D
        print('Dimension: 2D')
    elif len(input_shape) == 4:
        Conv = Conv3D
        ConvTranspose = Conv3DTranspose
        MaxPooling = MaxPooling3D
        print('Dimension: 3D')
        
    filter_sizes = [filtersize, 2 * filtersize, 4 * filtersize, 8 * filtersize, 16 * filtersize, 32 * filtersize]
    skipconnection = list()

    # Build encoder
    inputs = Input(shape=input_shape)
    encoder = inputs
    for i in range(depth):
        filtersize = filter_sizes[i]
        if net =='resnet':
            encoder = res_block(encoder,filtersize, input_shape)
        else:
            encoder = conv_block(encoder,filtersize, input_shape)

        skipconnection.append(encoder)
        if i < depth -1:
            encoder = MaxPooling(2)(encoder)
    
    # Build decoder
    decoder = encoder
    for i in range(depth-2,-1,-1):
        filtersize = filter_sizes[i]
        
        if AttGating:
            print('Using attention gating mechanism')
            skiptensor = att_block(skipconnection[i], decoder, filtersize*2, input_shape)
        else:
            skiptensor = skipconnection[i]
        
        decoder = ConvTranspose(filtersize, 2, strides=2, padding="same")(decoder)
        decoder = Concatenate()([decoder,skiptensor])
        if net=='resnet': 
            decoder = res_block(decoder,filtersize, input_shape)
        else:
            decoder = conv_block(decoder,filtersize, input_shape)
    
    # Last layer activation
    if n_classes == 1:      # Binary segmentation
        activation = 'sigmoid'
    else:                   # Multi-class segmentation
        activation = 'softmax'

    outputs = Conv(n_classes, 1, padding="same", activation=activation)(decoder)  # Activation based on n_classes
    print("Activation: ", activation)

    model = Model(inputs, outputs, name=net)
    return model