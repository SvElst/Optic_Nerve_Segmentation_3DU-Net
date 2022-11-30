#!/bin/bash

#Input data
SUBJID=$1       # Subject name
ARCHDIR=$2      # Data directory
DATADIR=$3      # Directory to save processed images
radius=$4       # Mask radius

# Subject directories
SUBJDIR=${ARCHDIR}/${SUBJID}
WORKDIR=${ARCHDIR}/${SUBJID}/preprocessing_files
SAVEDIR=${DATADIR}/${SUBJID}


# Check if the directory already exists
if [ -e ${SAVEDIR}/done2 ]
then
        echo "This subject is already being processed. Exiting"
        exit 2
fi

# If not, create it
mkdir ${WORKDIR}
mkdir ${SAVEDIR}
touch ${SAVEDIR}/done2


## Step 0. Reorient + Bias field correction
if [ ! -e ${WORKDIR}/fiesta.nii.gz ]
then
	echo "Reorient images"
        fslreorient2std ${SUBJDIR}/fiesta.nii ${WORKDIR}/fiesta
        fslreorient2std ${SUBJDIR}/gt_OD-label.nii.gz ${WORKDIR}/gt_OD-label
        fslreorient2std ${SUBJDIR}/gt_OS-label.nii.gz ${WORKDIR}/gt_OS-label

	echo "Bias field correction"
	N4BiasFieldCorrection -d 3 -i ${WORKDIR}/fiesta.nii.gz -o ${WORKDIR}/fiesta.nii.gz

fi

## Step 1. Detect eyes using Hough Transform
# Construct a rough segmentation on fiesta
./Preprocessing/create_Eyemasks_3DHough.sh ${WORKDIR}/fiesta true


## Step 2. Determine centroids of eyes
echo "Determine center points of eyes"
LABEL_1=(`cat ${WORKDIR}/fiesta_cluster_info.txt | sed -n 2p`)
LABEL_2=(`cat ${WORKDIR}/fiesta_cluster_info.txt | sed -n 3p`)

if (( $(echo "${LABEL_1[6]} < ${LABEL_2[6]}" | bc -l) ));
then
        pOSx=${LABEL_1[6]}
        pOSy=${LABEL_1[7]}
        pOSz=${LABEL_1[8]}
        pODx=${LABEL_2[6]}
        pODy=${LABEL_2[7]}
        pODz=${LABEL_2[8]}
else
        pOSx=${LABEL_2[6]}
        pOSy=${LABEL_2[7]}
        pOSz=${LABEL_2[8]}
        pODx=${LABEL_1[6]}
        pODy=${LABEL_1[7]}
        pODz=${LABEL_1[8]}

fi

## Step 3. Determine rotation angle and matrix to align eyes
echo "Calculate rotation matrix"
python3 ./Preprocessing/find_rotation_matrix.py -s $pOSx $pOSy -d $pODx $pODy -p ${WORKDIR}

## Step 4. Apply rotation matrix + resampling when angle > 5 degrees
echo "Apply transformation"
angle=(`cat ${WORKDIR}/angle.txt`)
cp ${WORKDIR}/angle.txt ${SAVEDIR}/angle.txt
if (( $(echo "$angle < 5" | bc -l) ));then
rotm=$FSLDIR/etc/flirtsch/ident.mat
echo "No rotation: angle=$angle"
else
rotm=${WORKDIR}/rotm.mat
echo "Rotation: angle=$angle"
fi

# Apply transformation to fiesta
if [ ! -e  ${SAVEDIR}/fiesta.nii.gz ];then
flirt -in ${WORKDIR}/fiesta.nii.gz -ref ${WORKDIR}/fiesta.nii.gz -out ${WORKDIR}/fiesta_rotated.nii.gz -applyisoxfm 0.3 -init $rotm -omat ${WORKDIR}/transform.mat -interp spline
cp ${WORKDIR}/fiesta_rotated.nii.gz ${SAVEDIR}/fiesta.nii.gz
cp ${WORKDIR}/transform.mat ${SAVEDIR}/transform.mat
fi

# Apply rotation matrix to determine new center voxels
if [ ! -e  ${SAVEDIR}/coordinates.nii.gz ];then
touch ${SAVEDIR}/coordinates.txt
echo "$pOSx $pOSy $pOSz" >> ${SAVEDIR}/coordinates.txt
echo "$pODx $pODy $pODz" >> ${SAVEDIR}/coordinates.txt

# Apply transformation to OS center voxels
sed -n 1p ${SAVEDIR}/coordinates.txt | img2imgcoord -src ${WORKDIR}/fiesta.nii.gz -dest ${WORKDIR}/fiesta_rotated.nii.gz -xfm ${WORKDIR}/transform.mat |sed -n 3p >> ${SAVEDIR}/coordinates.txt
# Apply transformation to OD center voxels
sed -n 2p ${SAVEDIR}/coordinates.txt | img2imgcoord -src ${WORKDIR}/fiesta.nii.gz -dest ${WORKDIR}/fiesta_rotated.nii.gz -xfm ${WORKDIR}/transform.mat |sed -n 3p >> ${SAVEDIR}/coordinates.txt
fi

# Apply transformation to GT
flirt -ref ${WORKDIR}/fiesta_rotated -in ${WORKDIR}/gt_OD-label -init ${WORKDIR}/transform.mat -applyxfm -out ${SAVEDIR}/gt_OD-label.nii.gz -interp nearestneighbour
flirt -ref ${WORKDIR}/fiesta_rotated -in ${WORKDIR}/gt_OS-label -init ${WORKDIR}/transform.mat -applyxfm -out ${SAVEDIR}/gt_OS-label.nii.gz -interp nearestneighbour

rm -r $WORKDIR
echo "Preprocessing pipeline completed"

exit

