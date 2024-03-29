# Optic nerve segmentation & quantification with 3D U-Net

This repository contains the code for optic nerve (ON) segmentation and quantification in MR-images. This code can be used to obtain 3D segmentations of the ON and cross-sectional measurements of the diameter and area along the lenght of the nerve.

## Content
- [Preprocessing](https://github.com/SvElst/Optic_Nerve_Segmentation_3DU-Net/tree/master/Preprocessing): Folder containing scripts to preprocess and crop input images.
  * [run_preprocessing.sh](https://github.com/SvElst/Optic_Nerve_Segmentation_3DU-Net/blob/master/Preprocessing/run_preprocessing.sh): Apply entire preprocessing pipeline and cropping to all subjects in data folder.
  * [preprocess.sh](https://github.com/SvElst/Optic_Nerve_Segmentation_3DU-Net/blob/master/Preprocessing/preprocess.sh): Preprocessing pipeline with reorientation, bias field correction, rotation and isotropic resampling.
  * [crop_images.sh](https://github.com/SvElst/Optic_Nerve_Segmentation_3DU-Net/blob/master/Preprocessing/crop_images.sh): Crop input image into two VOIs around eyes and ON. 
  * [create_Eyemasks_3DHough.sh](https://github.com/SvElst/Optic_Nerve_Segmentation_3DU-Net/blob/master/Preprocessing/create_Eyemasks_3DHough.sh): Eye and centroid detection using ITK 3D HoughTransform. 
  * [find_rotation_matrix.py](https://github.com/SvElst/Optic_Nerve_Segmentation_3DU-Net/blob/master/Preprocessing/find_rotation_matrix.py): Determine angle and rotation matrix between eye centroids


- [Segmentation](https://github.com/SvElst/Optic_Nerve_Segmentation_3DU-Net/tree/master/Segmentation): Folder containing scripts to train and test a U-Net for ON segmentation.
   * [augmentation.py](https://github.com/SvElst/Optic_Nerve_Segmentation_3DU-Net/blob/master/Segmentation/augmentation.py): Data augmentation
   * [DataGenerator.py](https://github.com/SvElst/Optic_Nerve_Segmentation_3DU-Net/blob/master/Segmentation/DataGenerator.py): Contains the class `DataGenerator` to generate batches of data
   * [load_data.py](https://github.com/SvElst/Optic_Nerve_Segmentation_3DU-Net/blob/master/Segmentation/load_data.py): Load images into memmory maps to restrict memmory consumption
   * [losses_metrics.py](https://github.com/SvElst/Optic_Nerve_Segmentation_3DU-Net/blob/master/Segmentation/losses_metrics.py): Includes various loss functions and performance evaluation metrics.
   * [predict.py](https://github.com/SvElst/Optic_Nerve_Segmentation_3DU-Net/blob/master/Segmentation/predict.py): Function to produce segmentations (as nifty) and performance results on specified evaluation metrics.
   * [test.py](https://github.com/SvElst/Optic_Nerve_Segmentation_3DU-Net/blob/master/Segmentation/test.py): Apply trained model to new data
   * [train.py](https://github.com/SvElst/Optic_Nerve_Segmentation_3DU-Net/blob/master/Segmentation/train.py): Train segmentation model
   * [unet_model.py](https://github.com/SvElst/Optic_Nerve_Segmentation_3DU-Net/blob/master/Segmentation/unet_model.py): Architecture of U-Net model. Adjusts to 2D or 3D structure dependent on input data. 


- [Quantification](https://github.com/SvElst/Optic_Nerve_Segmentation_3DU-Net/tree/master/Quantification)
    * [ON_Quantification_3DSlicer.py](https://github.com/SvElst/Optic_Nerve_Segmentation_3DU-Net/blob/master/Quantification/ON_quantification_3DSlicer.py): Usable in 3DSlicer Python Interface to extract diameter and cross-sectional area of each ON segmentation.

## Installation
Code is written in Python 3.6.9. 
The following dependencies are used: 
* Keras=2.1.0
* Tensorflow=2.2.0
* CUDA=10.1
* cuDNN=7.6.4
* Numpy = 1.19.5

Tools from [FSL](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki) and [ITK](https://itk.org/) were used in the preprocessing pipeline, which should be downloaded before usage. For quantification, [3D Slicer](https://www.slicer.org/) (version 5.0.3) is required.

## Usage

### Data structure
Data should be structured as follows:

    Data folder
    	 └── Subject1
			└── fiesta.nii.gz
			└── gt_OD-label.nii.gz 			# Right ON label
			└── gt_OS-label.nii.gz			# Left  ON label	        	
		 └── Subject2
		 └── etc.

### Preprocessing
To apply the entire preprocessing pipeline and cropping to all subjects in data folder, run the following command: 
```bash
bash ./Preprocessing/run_preprocess.sh PATH/TO/DATA/FOLDER
 ``` 
It requires the directory to the data folder as argument. 
Resultingly, two folders are created in the data folder: `Processed` contains preprocessed files per subject and their cropped images are saved in `Processed/Cropped`. Make sure the required ITK and FSL tools are downloaded and their paths (FSLDIR, ITKHOUGHDIR) specified correctly. 

### Segmentation
Train the segmentation model with: 
```python
python3 ./Segmentation/train.py --datadir <> --folderdir <> --output_prefix <> --inputsize <> --channels <> --classes <>
 ``` 

Test the segmentation model with:
```python
python3 ./Segmentation/test.py --datadir <> --folderdir <> --model <> --inputsize <> --channels <> --classes <>
 ``` 

**Required options**
* Data/save options:
	* --datadir = path to folder containing data
	* --folderdir = main working folder, in which output is saved
	* --output_prefix = name to save output (models, images,validation results and history) [training only]
    * --model = path to trained model [testing only]

* Input options:
	* --inputsize = size of iput image, should be dividable by two
	* --channels = sequence type (options: fiesta). Can be more than one when multiple sequences are used for segmentation. Subject folders should then include {sequence}.nii.gz file for each sequence. 
	* --classes = classes to segment, if>1: multiclass segmentation, otherwise: binary segmentation (options: Opticus, Eye, Tumor)
		* NOTE: Classes and its GT numerical values should be adapted in 'load_data.py' for each dataset accordingly

**Additional training options**
* Network options:
	* --net = type of network:
	    * unet = standard U-Net (default)
 	    * resnet = U-Net with residual blocks
	* --filtersize = number of base filters (default=32)
	* --depth = number of layers (default=5)
	* --attGate = use attention gating. 0=False,1=True (default=0)


* Training/Validation approach:
	* --nfolds = number of crossvalidation folds (default=10)
	    * if >1: k-fold crossvalidation
	    * if 0:  validation on hold-out test-set
	* --valindices = indices of subjects to be used for hold-out validation/testing
	    * NOTE: only applies if nfolds=0
	    * NOTE: repeat last index
		

* Learning options:
	* --lr = learning rate (default=0.0001)
	* --epochs = maximum number of epochs to train for (default=120)
	* --batch_size = number of scans per batch (default=1)


* Augmentation options:
	* --augment_flip = random left-right flipping probability (default=0.5)
	* --augment_scale = isotropic scaling [0.8-1.2] probability (default=0.5)
	* --augment_intensityShift= affine intensity shift [0.5-1.5] probability (default=0.3)
	* --augment_noise = additive Gaussian noise [sigma=0.1] probability (default=0.3)
	* --augment_blur = random Gaussian blurring probability (default=0.3)

**Additional testing options**
* --threshold = threshold to use for binarizing predicted segmentation(default=0.5)

**Data saving**

Data is saved at `Folderdir/Output/output_prefix/` 
* `images/` = contains the original image & GT & predicted segmentation for each validation/test subject
* `history/` = contains information of model training (loss/accuracy as function of epochs)
* `models/` = contains the saved model(s)
* `validation_results` = contains performance results for each evaluation metric on validation-set
* `test_results` = contains performance results for each evaluation metric on test-set


### Quantification
To obtain cross-sectional measurements along the length of each ON segmentation, run the file `ON_Quantification_3DSlicer.py` in 3DSlicer. 
Specify the path to the segmentations (containing original image and predicted segmentation of each subject) and the desired path to save the measurements. 

For each subject, a .CSV file is made containing the diameter and cross-sectional measurements.

