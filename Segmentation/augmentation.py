#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
On-the-fly date augmentation.
Using the Augmend package from: https://github.com/stardist/augmend

"""
import numpy as np
from augmend import Augmend, Identity, AdditiveNoise, IntensityScaleShift, IsotropicScale, Flip, GaussianBlur

def augment_4d(img, gt, augment_opts):        # Multichannel data
    aug = Augmend()
    
    # Transform both image and ground truth
    if augment_opts['Flip']:
        aug.add(list((Flip(axis=0),)*img.shape[-1])+[Flip(axis=0)], probability=augment_opts['Flip'])
    if augment_opts['Scale']:
        aug.add(list((IsotropicScale(axis=(0, 1, 2), amount=(0.8,1.2), order=3), )*img.shape[-1]) +
                  [IsotropicScale(axis=(0, 1, 2), amount=(0.8,1.2), order=0)],
                probability=augment_opts['Scale'])
    
    # Transform only image
    if augment_opts['Intensity_shift']:
        aug.add(list((IntensityScaleShift(scale=(0.5, 1.5)), )*img.shape[-1])+ [Identity()], probability=augment_opts['Intensity_shift'])
    if augment_opts['Additive_noise']:
        aug.add(list((AdditiveNoise(sigma=0.1),)*img.shape[-1]) +[Identity()], probability=augment_opts['Additive_noise'])
    if augment_opts['Gaussian_blur']:
        aug.add(list((GaussianBlur(amount=(0.2,1.2)),)*img.shape[-1]) +[Identity()], probability=augment_opts['Gaussian_blur'])
    
    if img.shape[-1] == 1:  # One channel
        img_aug, gt_aug = aug([img[:,:,:,0], gt])
        img_aug=np.expand_dims(img_aug, -1)
    if img.shape[-1] == 2:  # Multiple channels
        img_aug1, img_aug2, gt_aug =  aug([img[:,:,:,0], img[:,:,:,1],gt ])
        img_aug1=np.expand_dims(img_aug1, -1)
        img_aug2=np.expand_dims(img_aug2, -1)
        img_aug = np.concatenate([img_aug1, img_aug2], -1)
    return img_aug, gt_aug


