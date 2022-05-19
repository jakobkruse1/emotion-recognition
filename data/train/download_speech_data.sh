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

function ProgressBar {
# Process data
    let _progress=(${1}*100/${2}*100)/100
    let _done=(${_progress}*4)/10
    let _left=40-$_done
# Build progressbar string lengths
    _fill=$(printf "%${_done}s")
    _empty=$(printf "%${_left}s")

# 1.2 Build progressbar strings and print the ProgressBar line
# 1.2.1 Output example:
# 1.2.1.1 Progress : [########################################] 100%
printf "\rProgress : [${_fill// /#}${_empty// /-}] ${_progress}%%"

}

if [ ! -f "meld/done" ]; then
  set -e
  if [ ! -d "meld" ]; then
    mkdir meld
    wget -nc http://web.eecs.umich.edu/~mihalcea/downloads/MELD.Raw.tar.gz
    tar -xf MELD.Raw.tar.gz -C meld
    rm -f MELD.Raw.tar.gz
    mv meld/MELD.Raw/* meld/
    rm -rf meld/MELD.Raw
  fi
  cd meld
  if [ ! -d "output_repeated_splits_test" ]; then
    tar -xf train.tar.gz
    tar -xf dev.tar.gz
    tar -xf test.tar.gz
    rm -f output_repeated_splits_test/._dia*
    rm ._output_repeated_splits_test
    rm dev.tar.gz
    rm test.tar.gz
    rm train.tar.gz
  fi
  mkdir -p train
  mkdir -p val
  mkdir -p test
  rm -f train/*.wav
  rm -f val/*.wav
  rm -f test/*.wav
  set +e
  num_files=$(ls train_splits | wc -l)
  index=0
  printf "\nConverting train mp4 files to audio files. This may take a while!\n"
  rm -f train_splits/dia125_utt3.mp4  # Broken file
  for f in train_splits/*.mp4; do
    ffmpeg -hide_banner -loglevel error -i ${f} train/${f:13}.wav
    index=$(($index+1))
    ProgressBar ${index} ${num_files}
  done
  num_files=$(ls dev_splits_complete | wc -l)
  index=0
  printf "\nConverting validation mp4 files to audio files. This may take a while!\n"
  for f in dev_splits_complete/*.mp4; do
    ffmpeg -hide_banner -loglevel error -i ${f} val/${f:20}.wav
    index=$(($index+1))
    ProgressBar ${index} ${num_files}
  done
  num_files=$(ls output_repeated_splits_test | wc -l)
  index=0
  printf "\nConverting test mp4 files to audio files. This may take a while!\n"
  for f in output_repeated_splits_test/*.mp4; do
    ffmpeg -hide_banner -loglevel error -i ${f} test/${f:28}.wav
    index=$(($index+1))
    ProgressBar ${index} ${num_files}
  done
  rm -rf train_splits
  rm -rf output_repeated_splits_test
  rm -rf dev_splits_complete
  cd ..
  touch meld/done
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
