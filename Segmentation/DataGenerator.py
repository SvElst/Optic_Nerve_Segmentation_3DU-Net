#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate data in batches

"""
import numpy as np
from tensorflow.python import keras
from keras.utils import to_categorical
from augmentation import augment_4d

class DataGenerator(keras.utils.data_utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, d_memmap, batch_size=1, dim=(144,144,96), channels=['fiesta'], n_classes=6, shuffle=True, mode = 'train', augment_opts={}):
        'Initialization', 
        self.list_IDs = list_IDs
        self.d_memmap = d_memmap
        self.batch_size = batch_size
        self.dim = dim
        self.channels = channels
        self.n_channels = len(channels)
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.mode = mode
        self.on_epoch_end()
        self.augment_opts=augment_opts

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)
           
        if self.mode == 'test':
            return X
        else:
            return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing #batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.zeros((self.batch_size, *self.dim, self.n_channels), dtype=np.float32)
        y = np.zeros((self.batch_size, *self.dim), dtype=np.short)
        
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            ch = np.zeros((self.dim[0], self.dim[1], self.dim[2], self.n_channels))
            for j in range(self.n_channels):
                tmp = self.d_memmap[self.channels[j]][ID]
                ch[:,:,:,j] = tmp

            # Store sample
            X[i,] = ch              
    
            # Store label
            if self.mode != 'test':
                y[i,] = self.d_memmap['gt'][ID]
                
            # Data augmentation                  
            if self.augment_opts:                # Only in train fase
                X[i,],y[i,] = augment_4d(X[i,], y[i,], self.augment_opts)
                        
        if self.n_classes==1:   # Binary segmentation
            return X.astype(np.float32), y.astype(np.short)

        return X.astype(np.float32), to_categorical(y.astype(np.short), num_classes=self.n_classes)