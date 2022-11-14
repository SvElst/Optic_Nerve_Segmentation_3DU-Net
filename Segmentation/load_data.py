#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Load images into memmory maps to restrict memmory consumption

"""
import nibabel as nib
import numpy as np


"Function to normalize entire patch" 
def norm(input_img):
    input_mean = input_img.mean()
    input_std = input_img.std()
    input_img -= input_mean
    input_img /= input_std 
    return input_img

#%%
def load(SUBJIDS,datadir,nclasses,classes, channels, patch_size, memmap_path, fp_shape):
    imvar = 0
    subjid_list = []
    leftright=['OD', 'OS']

    for subjid in SUBJIDS:
        for lr in leftright:
            print('Subject: ', subjid, 'Hemi: ', lr)
            print('Loading GT...')
            try:
                gt = np.asanyarray(nib.load(datadir + subjid + '/gt_'+ lr + '-label.nii.gz').dataobj).astype(np.short)
            except:
                print('No label ' + lr)
                continue        
            print('')
                    
            "Define type of segmentation and convert GTs accordingly (depends on numbering of classes in GT)"
            # Binary segmentation
            if nclasses==1:                 
                if classes == ['Opticus']: 
                    gt[gt != 6] = 0     
                    gt[gt == 6] = 1
                if classes == ['Tumor']: 
                    gt[gt == 9] = 5
                    gt[gt != 5] = 0  
                    gt[gt == 5] = 1  
                if classes == ['Eye']:                  
                    gt[gt == 6] = 0
                    gt[gt != 0] = 1 
            else:
            # Multiclass segmentation
                gt[gt == 2] = 1
                gt[gt == 3] = 1 
                gt[gt == 4] = 1 
                gt[gt == 7] = 1     
                gt[gt == 9] = 2
                gt[gt == 5] = 2     
                if 'Opticus' in classes:
                    gt[gt == 6] = 3     # Include opticus as class
                    if 'Eye' not in classes:    
                     gt[gt == 1] = 0
                     gt[gt == 2] = 1
                     gt[gt == 3] = 2
                else:  
                    gt[gt == 6] = 0     # Exclude opticus as class  
                
            # Load images per channel/sequence
            for channel in channels:
                print('Loading {}...'.format(channel))
                scan_nii = nib.load(datadir + subjid + '/{}_'.format(channel) + lr +'.nii.gz') 
                aff = scan_nii.get_affine()
                head = scan_nii.get_header()
                pixdim = [scan_nii.header['pixdim'][1], scan_nii.header['pixdim'][2], scan_nii.header['pixdim'][3]]
                scan = scan_nii.get_fdata().astype(np.float32)
                    
                print('Normalization')
                scan= norm(scan)
                print('')
                
                # Save image to memmory map
                fp_images = np.memmap(memmap_path + 'memmaps/images_{}.dat'.format(channel), dtype=np.float32, mode='r+', shape=fp_shape)
                fp_images[imvar] = scan
                del fp_images, scan    # delete variable 
    
            # Save gt to memmory map
            fp_images_gt = np.memmap(memmap_path + 'memmaps/images_gt.dat', dtype=np.short, mode='r+', shape=fp_shape)
            fp_images_gt[imvar,]=gt
            del fp_images_gt, gt
    
            imvar += 1
               
            # Save subject info in dict           
            dictd = {'subjid': subjid, 'hemi': lr, 'aff': aff, 'head': head, 'pixdim': pixdim}    
            subjid_list.append(dictd)
            
    return subjid_list