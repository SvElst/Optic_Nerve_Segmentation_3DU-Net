#!/bin/bash

# This file executes the preprocessing pipeline for every patient in the directory.

if [ "$#" -lt 1 ]   # $# = number of items on command line
then
        echo "Usage: Enter path to data directory"
        exit 2
fi

DATADIR=$1
SAVEDIR=${DATADIR}/Processed
# Make directory to save processed images (if directory does not yet exist)
if [ ! -d ${SAVEDIR} ]
then
    mkdir $SAVEDIR
fi

CROPDIR=${DATADIR}/Processed/Cropped
# Make directory to save cropped images (if directory does not yet exist)
if [ ! -d ${CROPDIR} ]
then
    mkdir $CROPDIR
fi

##
# Run a job for every Patient
##
radius=6
for DIR in $DATADIR/Subject*
do
        SUBJID=${DIR##/*/}

        echo "Start preprocessing pipeline - " ${SUBJID}
        ./Preprocessing/preprocess.sh ${SUBJID} ${DATADIR} ${SAVEDIR} $radius

        echo "Crop image - " ${SUBJID}     
        ./Preprocessing/crop_images.sh ${SUBJID} ${SAVEDIR} ${CROPDIR};
done;
