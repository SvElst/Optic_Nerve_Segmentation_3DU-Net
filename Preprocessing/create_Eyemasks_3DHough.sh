#!/bin/bash
# Apply ITK 3D Hough transform to detect circles in the input image. Circles represent rough segmentation of the eyes.
# Centroids are extracted from these circles

# Remove nii.gz from image file name
I=`${FSLDIR}/bin/remove_ext $1`

# Subsample image by 2
fslmaths ${I} -subsamp2 ${I}_subsamp2

# Detect eyes
if [ "$2" == "true" ]; then
        # Invert FIESTA and T2
        MAX_I=(`fslstats ${I}_subsamp2 -R`)
        fslmaths ${I}_subsamp2 -mul -1 -add ${MAX_I[1]} ${I}_subsamp2_inv
        # Use 3D Hough transform to detect circle with radius between 9-15 mm
    	echo "Create Hough Map"
        ${ITKHOUGHDIR} ${I}_subsamp2_inv.nii.gz ${I}_accum.nii.gz ${I}_subsamp2_seg.nii.gz 2 9 15 1 1 1 0.5 0 0 0 64 0.1 > /dev/null
else
        # Do not invert T1c
    	echo "Create Hough Map"
    	${ITKHOUGHDIR} ${I}_subsamp2.nii.gz ${I}_accum.nii.gz ${I}_subsamp2_seg.nii.gz 2 9 15 1 1 1 0.5 0 0 0 64 0.1 > /dev/null
fi


# Fill detected circles
fslmaths ${I}_subsamp2_seg -fillh  ${I}_subsamp2_seg_filled

# Upsample mask with two (using registration with identity matrix)
flirt -in ${I}_subsamp2_seg_filled -ref ${I} -out ${I}_seg_filled -applyxfm -interp nearestneighbour -init $FSLDIR/etc/flirtsch/ident.mat

## Separate the two eyes.
cluster -i ${I}_seg_filled.nii.gz -t 0.1 -o ${I}_seg_filled_cluster_index > ${I}_cluster_info.txt

## Find centroids
LABEL_1=(`cat ${I}_cluster_info.txt | sed -n 2p`)
LABEL_2=(`cat ${I}_cluster_info.txt | sed -n 3p`)

echo "Circle 1 center x: ${LABEL_1[6]}"
echo "Circle 2 center x: ${LABEL_2[6]}"

exit

