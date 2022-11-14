#!/bin/bash

SUBJID=$1           # Subject name
DATADIR=$2          # Directory to processed data
WORKDIR=$3          # Directory to save cropped images

SUBDIR=${DATADIR}/$SUBJID       # Current subject directory

# Exit if subject is already cropped
if [ -d ${WORKDIR}/$SUBJID ];then
	echo "This subject is already cropped"
	exit
fi

# If not, create folder
mkdir $WORKDIR/$SUBJID

# Get center points of eyes in rotated image
pOS=(`cat ${SUBDIR}/coordinates.txt | sed -n 3p`)
pOD=(`cat ${SUBDIR}/coordinates.txt | sed -n 4p`)

# Round pixel coordinates
OSx=(`printf "%.0f\n" ${pOS[0]}`)
OSy=(`printf "%.0f\n" ${pOS[1]}`)
OSz=(`printf "%.0f\n" ${pOS[2]}`)
ODx=(`printf "%.0f\n" ${pOD[0]}`)
ODy=(`printf "%.0f\n" ${pOD[1]}`)
ODz=(`printf "%.0f\n" ${pOD[2]}`)

# Centerpoint between two eyes
Cx=$( echo "($ODx - $OSx)/2 + $OSx" | bc -l )
Cx=(`printf "%.0f\n" $Cx`)
Cy=${OSy}

z=(`fslinfo ${SUBDIR}/fiesta.nii.gz | sed -n 4p`)
z=${z[1]}
Cz=$(( $z / 2 ))

# Minimum values of bounding box
zmin=$(( $Cz - 56))
ymin_OS=$(( OSy - 170 ))
ymin_OD=$(( ODy - 170 ))
xmin_OS=$(( $Cx - 144))
xmin_OD=$Cx

# Crop scans to VOIs
fslroi ${SUBDIR}/fiesta.nii.gz ${WORKDIR}/$SUBJID//fiesta_OS.nii.gz $xmin_OS 144 $ymin_OS 240 $zmin 112 
fslroi ${SUBDIR}/fiesta.nii.gz ${WORKDIR}/$SUBJID/fiesta_OD.nii.gz $xmin_OD 144 $ymin_OD 240 $zmin 112 

# Crop GT to VOI
fslroi ${SUBDIR}/gt_OS-label.nii.gz ${WORKDIR}/$SUBJID/gt_OS-label.nii.gz $xmin_OS 144 $ymin_OS 240 $zmin 112 
fslroi ${SUBDIR}/gt_OD-label.nii.gz ${WORKDIR}/$SUBJID/gt_OD-label.nii.gz $xmin_OD 144 $ymin_OD 240 $zmin 112 
