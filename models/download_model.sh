#!/bin/bash

cd $(dirname "$BASH_SOURCE")

POSITIONAL_ARGS=()

while [[ $# -gt 0 ]]; do
  case $1 in
    -d|--data)
      MODALITY="$2"
      shift # past argument
      shift # past value
      ;;
    -m|--model)
      MODEL="$2"
      shift # past argument
      shift # past value
      ;;
    -h|--help)
      echo "Script that is used to download trained models for all modalities."
      echo "Both the 'data' and the 'model' parameter must be given!"
      echo "Arguments:"
      echo " -d|--data:   Data type or modality of the model."
      echo "        Options: image, text, speech, plant, watch"
      echo " -m|--model:  Name of the model to download."
      echo "        Options:"
      echo "            image  -> vgg16"
      echo "            text   -> distilbert"
      echo "            speech -> hubert"
      echo "            plant  -> mfcc_resnet"
      echo "            watch  -> random_forest"
      exit 0
      ;;
    -*|--*)
      echo "Unknown option $1"
      exit 1
      ;;
    *)
      POSITIONAL_ARGS+=("$1") # save positional arg
      shift # past argument
      ;;
  esac
done

set -- "${POSITIONAL_ARGS[@]}" # restore positional parameters

echo "MODALITY  = ${MODALITY}"
echo "MODEL     = ${MODEL}"

if [[ -z ${MODALITY} ]] || [[ -z ${MODEL} ]]; then
  echo "Both the --data and -- model parameters must be given! Aborting."
  echo "For help, use the -h or --help flags on this script."
  exit 1
fi

check_folder () {
  FOLDER=$1
  # Check if folder exists and model is there.
  if [ -d ${FOLDER} ]; then
    echo "The model folder '${FOLDER}' exists. Please delete the folder to continue."
    exit 1
  fi
  # Create folder if it does not exist
  PARENTDIR="$(dirname "${FOLDER}")"
  if [ ! -d ${PARENTDIR} ]; then
    mkdir ${PARENTDIR}
  fi
}

if [ ${MODALITY} == "image" ]; then
  if [ ${MODEL} == "vgg16" ]; then
    # Download model
    check_folder "image/vgg16"
    MODEL_PATH=https://www.dropbox.com/s/dey7m0dvlsii86x/image_vgg16.zip
    wget ${MODEL_PATH}
    unzip image_vgg16.zip -d image
    rm image_vgg16.zip
  else
    echo "Model ${MODEL} is not available in the image modality. Use --help for details."
    exit 1
  fi;
elif [ ${MODALITY} == "text" ]; then
  if [ ${MODEL} == "distilbert" ]; then
    # Download model
    check_folder "text/distilbert"
    MODEL_PATH=https://www.dropbox.com/s/g1iffe9t0lqimkb/text_distilbert.zip
    wget ${MODEL_PATH}
    unzip text_distilbert.zip -d text
    rm text_distilbert.zip
  else
    echo "Model ${MODEL} is not available in the text modality. Use --help for details."
    exit 1
  fi;
elif [ ${MODALITY} == "speech" ]; then
  if [ ${MODEL} == "hubert" ]; then
    # Download model
    check_folder "speech/hubert"
    MODEL_PATH=https://www.dropbox.com/s/fxgmua6y1nallzc/speech_hubert.zip
    wget ${MODEL_PATH}
    unzip speech_hubert.zip -d speech
    rm speech_hubert.zip
  else
    echo "Model ${MODEL} is not available in the speech modality. Use --help for details."
    exit 1
  fi;
elif [ ${MODALITY} == "plant" ]; then
  if [ ${MODEL} == "mfcc_resnet" ]; then
    # Download model
    check_folder "plant/plant_mfcc_resnet_0"
    check_folder "plant/plant_mfcc_resnet_1"
    check_folder "plant/plant_mfcc_resnet_2"
    check_folder "plant/plant_mfcc_resnet_3"
    check_folder "plant/plant_mfcc_resnet_4"
    MODEL_PATH=https://www.dropbox.com/s/of8swm2aer8zw4r/plant_mfcc_resnet.zip
    wget ${MODEL_PATH}
    unzip plant_mfcc_resnet.zip -d plant
    rm plant_mfcc_resnet.zip
  else
    echo "Model ${MODEL} is not available in the plant modality. Use --help for details."
    exit 1
  fi;
elif [ ${MODALITY} == "watch" ]; then
  if [ ${MODEL} == "random_forest" ]; then
    # Download model
    check_folder "watch/random_forest_0"
    check_folder "watch/random_forest_1"
    check_folder "watch/random_forest_2"
    check_folder "watch/random_forest_3"
    check_folder "watch/random_forest_4"
    MODEL_PATH=https://www.dropbox.com/s/feo6d40uet9swnq/watch_random_forest.zip
    wget ${MODEL_PATH}
    unzip watch_random_forest.zip -d watch
    rm watch_random_forest.zip
  else
    echo "Model ${MODEL} is not available in the watch modality. Use --help for details."
    exit 1
  fi;
else
  echo "Modality ${MODALITY} is not available. Use --help for details."
  exit 1
fi;
