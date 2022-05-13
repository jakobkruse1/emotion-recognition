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
  rm -rf speech/test
  rm -rf speech/train
  rm -rf speech/val
fi

if [ ! -d "speech" ]; then
  mkdir speech
fi

cd speech

# Download ravdess dataset
if [ ! -d "ravdess" ] || [ $FORCE -eq 1 ]; then
  wget -nc https://zenodo.org/record/1188976/files/Audio_Speech_Actors_01-24.zip
  unzip Audio_Speech_Actors_01-24.zip -d ravdess
  rm Audio_Speech_Actors_01-24.zip
fi


# Prepare all the data
cd ../../..
if [ -f "data/train/speech/done" ] && [ "${FORCE}" -eq 0 ]; then
    echo "Folder exists. Skipping. Use '-f' to force re-downloading."
else
    set -e
    rm -rf data/train/speech/train
    rm -rf data/train/speech/val
    rm -rf data/train/speech/test
    if [ -f "venv/bin/python" ]; then
        venv/bin/python data/train/prepare_speech_data.py
    else
        # In CI or docker there is no venv
        export PYTHONPATH=.
        python data/train/prepare_speech_data.py
    fi
    touch data/train/speech/done
fi
