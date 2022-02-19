# Data Folder Structure

This file describes how you need to setup the data folder for the code to work properly.
This folder contains two subfolders:
The train folder is used to store all the data that is used for training the emotion classification models.
The eval folder contains the custom data that is used for evaluation accross different modalities.

## Training data
To download the training datasets required for training certain emotion classification models,
use the bash scripts provided in the train folder.

### Text Data
To download the text datasets, please use the `download_text_data.sh` script in the train folder.
This will download two datasets:
1. [Emotions dataset from huggingface](https://huggingface.co/datasets/emotion)
2. [GoEmotions dataset from google research](https://github.com/google-research/google-research/tree/master/goemotions)

After downloading, the datasets will be combined and stored in the text subfolder.

## Eval data
TODO
