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
      FORCE=1
      ;;
    \?)
      echo "Unexpected option -$OPTARG"
      ;;
  esac
done

if [ ! -d "text" ]; then
  mkdir text
fi

# Download three datasets
cd text

## Download emotions dataset
_TRAIN_DOWNLOAD_URL="https://www.dropbox.com/s/1pzkadrvffbqw6o/train.txt?dl=1"
_VALIDATION_DOWNLOAD_URL="https://www.dropbox.com/s/2mzialpsgf9k5l3/val.txt?dl=1"
_TEST_DOWNLOAD_URL="https://www.dropbox.com/s/ikkqxfdbdec3fuj/test.txt?dl=1"
wget -nc -O train.txt $_TRAIN_DOWNLOAD_URL
wget -nc -O validation.txt $_VALIDATION_DOWNLOAD_URL
wget -nc -O test.txt $_TEST_DOWNLOAD_URL

## Download go-emotions dataset
wget -nc -O emotions.txt https://raw.githubusercontent.com/google-research/google-research/master/goemotions/data/emotions.txt
wget -nc -O dev.tsv https://raw.githubusercontent.com/google-research/google-research/master/goemotions/data/dev.tsv
wget -nc -O train.tsv https://raw.githubusercontent.com/google-research/google-research/master/goemotions/data/train.tsv
wget -nc -O test.tsv https://raw.githubusercontent.com/google-research/google-research/master/goemotions/data/test.tsv

# Download crowdflower dataset - unsure if it will be used because not cleaned
wget -nc -O emotions.csv https://query.data.world/s/5tkwav7lbu3v3l2rgwpuu3sfjrorti

# Combine them into one csv
cd ../../..
venv/bin/python data/train/combine_text_data.py
