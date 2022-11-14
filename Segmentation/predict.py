#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Obtain segmentation predictions for each subject. 
Saves predictions as images and evaluation metrics. 

"""
import numpy as np
import pandas as pd
import nibabel as nib
from losses_metrics import spatial_metrics
from losses_metrics import distance_metrics

def predict(model, generator,val_indices,  d_memmap, nclasses, subjid_list, path, result_path, threshold=0.5, save_images=False):

    #Predict on the test data
    y_pred=model.predict(generator, verbose=1)      # Predicted segmentation
    if nclasses >1:     # = Multiclass segmentation
        y_pred_argmax=np.argmax(y_pred, axis=4)
    else:               # = Binary segmentation
        threshold=threshold  
        y_pred_argmax=(y_pred>threshold).astype(np.uint8)
    y_gt_argmax = d_memmap['gt']                    # Ground truth segmentation
        
    # Initialize evaluation metrics
    metrics = ['dice', 'vol', 'vol_ref', 'vol_rel', 'precision', 'recall', 'aSD', 'hd95']
    eval_metric = dict((i,np.zeros((len(y_pred_argmax),nclasses))) for i in metrics) 
    
    df = pd.DataFrame(columns = ['Subject', 'Hemi'])                                      
    for p, index in enumerate(val_indices):   # For each subject
        df.loc[p] = [subjid_list[index]['subjid'], subjid_list[index]['hemi']] 
        for c in range(nclasses):
            if nclasses==1:
                c = nclasses
                
            im_gt = np.asarray(y_gt_argmax[index] == c).astype(np.bool)    
            im_pred = np.asarray(y_pred_argmax[p] == c).astype(np.bool)
            im_pred = np.resize(im_pred,im_gt.shape)       
            
            if nclasses==1:
                c = 0     
                
            pixdim = subjid_list[index]['pixdim']          
            metric_values = spatial_metrics(im_gt, im_pred, pixdim[0], pixdim[1], pixdim[2]) + distance_metrics(im_gt, im_pred)
            for v, metric in enumerate(metrics):
                eval_metric[metric][p,c] = metric_values[v]

        if save_images:
            seg_nii = nib.Nifti1Image(y_pred_argmax[p], subjid_list[index]['aff'],subjid_list[index]['head']) 
            seg_nii.header.set_data_dtype(np.float32)
            nib.save(seg_nii,path + 'images/' + subjid_list[index]['subjid'] + '_' + subjid_list[index]['hemi'] + '-label.nii.gz')
            gt_nii = nib.Nifti1Image(y_gt_argmax[index], subjid_list[index]['aff'],subjid_list[index]['head']) 
            gt_nii.header.set_data_dtype(np.float32)
            nib.save(gt_nii,path + 'images/' + subjid_list[index]['subjid'] + '_' + subjid_list[index]['hemi'] + '_gt-label.nii.gz')
            im_nii = nib.Nifti1Image(d_memmap['fiesta'][index], subjid_list[index]['aff'],subjid_list[index]['head']) 
            im_nii.header.set_data_dtype(np.float32)
            nib.save(im_nii,path + 'images/' + subjid_list[index]['subjid'] + '_' + subjid_list[index]['hemi'] + '-image.nii.gz')
    
    # Save metrics in csv files
    for metric in metrics:
        df_metric = pd.concat([df, pd.DataFrame(np.around(eval_metric[metric], decimals=3))], axis=1)
        df_metric.to_csv(result_path + f'{metric}_all_subjects.csv', mode='w', header=False) 
