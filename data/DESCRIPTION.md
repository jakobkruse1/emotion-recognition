# Data Folder Structure

This file describes how you need to setup the data folder for the code to work properly.
This folder contains two subfolders:
The train folder is used to store all the data that is used for training the emotion classification models.
The eval folder contains the custom data that is used for evaluation accross different modalities.

*When using any of these datasets, please remember to cite the corresponding papers!*

## Training data
To download the training datasets required for training certain emotion classification models,
use the bash scripts provided in the train folder.

### Text Data
To download the text datasets, please use the `download_text_data.sh` script in the train folder.
This will download two datasets:
1. [Emotions dataset from huggingface](https://huggingface.co/datasets/emotion)
2. [GoEmotions dataset from google research](https://github.com/google-research/google-research/tree/master/goemotions)

After downloading, the datasets will be combined and stored in the text subfolder.

### Facial Data
To download the text datasets, please use the `download_image_data.sh` script in the train folder.
This will download two datasets:
1. [FER2013+ Dataset](https://github.com/microsoft/FERPlus)
   1. This dataset requires a manual download step. The FER2013 dataset needs to be downloaded manually.
   2. Go to [this website](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data) and download the fer2013.tar.gz file.
   3. Create the folder data/train/image/fer2013
   4. Copy the file fer2013.csv from fer2013.tar.gz to the created folder
   5. After that, run the `download_image_data.sh` script.
2. [Kaggle Dataset](https://www.kaggle.com/datasets/jonathanoheix/face-expression-recognition-dataset)
   1. This requires you to [setup the kaggle API](https://www.kaggle.com/docs/api) first and create a kaggle account.
   2. After that, run the `download_image_data.sh` script.
3. [Jaffe Dataset](https://zenodo.org/record/3451524#.YlRNsTzb1H4)
   1. You need to download this dataset manually. Go to the link above and request access to the data.
   2. Download the data and put the zip file in the `data/train/images` folder.
   3. After that, run the `download_image_data.sh` script.
4. [CK+ Dataset](https://paperswithcode.com/dataset/ck)
   1. This data needs to be downloaded manually. Go to the link above and request the data.

## Eval data
TODO
