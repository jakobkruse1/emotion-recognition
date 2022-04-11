#!/bin/bash

cd $(dirname "$BASH_SOURCE")

# Setup
DOCKER_CONFIG=0
FORCE=0
while getopts ":df" opt; do
  case $opt in
    d)
      # If -d flag is specified, run installation without sudo
      echo "Using docker configuration without sudo!"
      DOCKER_CONFIG=1
      ;;
    f)
      # Force flag that overwrites data also if it is already there
      echo "Forcing download of the required files!"
      FORCE=1
      ;;
    \?)
      echo "Unexpected option -$OPTARG"
      ;;
  esac
done

if [ $FORCE -eq 1 ]; then
  echo "Force"
fi

if [ ! -d "image" ]; then
  mkdir image
fi

# Download three datasets
cd image

# Download FER2013 dataset
# This dataset needs to be downloaded manually. The data can be found at:
# https://www.kaggle.com/competitions/challenges-in-representation-learning-facial-expression-recognition-challenge/data
# The only file that is required for this code to run is the
# fer2013.csv file inside fer2013.tar.gz .
# Please create the folder "fer2013" under data/train/image/fer2013 and save the fer2013.csv file in this folder.
if [ ! -f "fer2013/fer2013.csv" ]; then
    echo "The FER2013 dataset needs to be downloaded manually. See the DESCRIPTION.md for more details."
else
    wget -nc -O fer2013/fer_labels.csv https://raw.githubusercontent.com/microsoft/FERPlus/master/fer2013new.csv
fi;

# Download kaggle dataset
if [ ! -d "images" ]; then
    kaggle datasets download -d jonathanoheix/face-expression-recognition-dataset
    unzip face-expression-recognition-dataset.zip
    rm -rf images/images
    rm face-expression-recognition-dataset.zip
fi;

# Extract JAFFE dataset if it exists
if [ -f "jaffedbase.zip" ]; then
    unzip jaffedbase.zip
    rm -rf __MACOSX
    rm jaffedbase.zip
fi;


# Prepare all the data
cd ../../..
if [ -f "data/train/image/done" ] && [ "${FORCE}" -eq 0 ]; then
    echo "CSV files exists. Skipping"
else
    rm -rf data/train/image/train
    rm -rf data/train/image/val
    rm -rf data/train/image/test
    if [ -f "venv/bin/python" ]; then
        venv/bin/python data/train/prepare_image_data.py
    else
        # In CI or docker there is no venv
        export PYTHONPATH=.
        python data/train/prepare_image_data.py
    fi
fi
