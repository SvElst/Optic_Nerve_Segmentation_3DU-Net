#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Loss functions and spatial, distance and volumetric evaluation metrics
"""

# Import packages 
from tensorflow.python.keras import backend as K
from tensorflow.python import keras
import numpy as np
import sys
# ICC
import pingouin as pg 
import pandas as pd
# Surface distance: https://github.com/deepmind/surface-distance
import surface_distance
#%% Loss functions
def dice_coefficient(y_true, y_pred):
    smoothing_factor = 1
    flat_y_true = K.flatten(y_true)
    flat_y_pred = K.flatten(y_pred)
    return (2. * K.sum(flat_y_true * flat_y_pred) + smoothing_factor) / (K.sum(flat_y_true) + K.sum(flat_y_pred) + smoothing_factor)

def dice_coefficient_loss(y_true, y_pred):
    return 1 - dice_coefficient(y_true, y_pred)

#%% Evaluation metrics
" Spatial evaluation metrics" 
def spatial_metrics(gt, pred, deltax, deltay, deltaz):
    intersection = np.logical_and(gt, pred)
    dice = 2.*intersection.sum() / (gt.sum()+pred.sum())
    vol = pred.sum()*deltax*deltay*deltaz
    vol_ref = gt.sum()*deltax*deltay*deltaz
    try:
        vol_rel = vol/vol_ref
    except:
        vol_rel = 0
    try:
        Precision = intersection.sum() / pred.sum()
    except:
        Precision = 0
    try:
        Recall = intersection.sum() / gt.sum()
    except:
        Recall = 0
    try:
        IOU = intersection.sum() / (gt.sum() + gt.sum() - intersection.sum())
    except:
        IOU=0
    def rounding(metric):
        metric = np.around(metric, decimals=3)
        return metric
    return rounding(dice), rounding(vol), rounding(vol_ref), rounding(vol_rel), rounding(Precision), rounding(Recall)

"Distance evaluation metrics"
def distance_metrics(im_gt, im_pred, dim=0.3):
    surface_distances = surface_distance.compute_surface_distances(im_gt, im_pred, spacing_mm=(dim, dim, dim))
    aSD=surface_distance.compute_average_surface_distance(surface_distances)                                    # Average surface distance
    hd95 = surface_distance.compute_robust_hausdorff(surface_distances, 95)                                     # Hausdorff distance 95%        
    dice = surface_distance.compute_dice_coefficient(im_gt, im_pred)                                            # Dice
    dice_tol1voxel = surface_distance.compute_surface_dice_at_tolerance(surface_distances, tolerance_mm=dim)    # Dice with tolerance to one voxel

    return (aSD[0]+aSD[1])/2, hd95

"Volumetric evaluation metric: ICC "
def ICC(df_vol, df_volref):
    df_vol = pd.concat([df_vol, pd.Series([0,]*len(df_vol))], axis=1)
    df_volref = pd.concat([df_volref, pd.Series([1,]*len(df_volref))], axis=1)
    df_vol_all = pd.concat([df_vol, df_volref.rename(columns={'Vol_ref':'Vol'})], ignore_index=True)
    df_vol_all = df_vol_all.set_axis(['Subject', 'Hemi', 'Volume', 'Method'], axis=1, inplace=False)
    icc = pg.intraclass_corr(data=df_vol_all, targets='Subject', raters='Method', ratings='Volume')
    return icc
