# Data Folder Structure

This file describes how you need to setup the data folder for the code to work properly.
This folder contains multiple subfolders:
The train folder is used to store all the data that is used for training the text, voice and face emotion classification models.
The other folders contain the custom data that is used for evaluation across different modalities.
<div class="disclaimer" style="background-color: #f5ea92">
   ⚠️&nbsp When using any of these datasets, please remember to cite the corresponding papers! Check the websites for more information.
</div>

## Training data
To download the training datasets required for training certain emotion classification models,
use the bash scripts provided in the train folder.  
Some of the datasets can not be downloaded automatically because you need to request access to them and they are to be used for research only.
The following sections will list all the datasets and how you can access them.

### Text Data
To download the text datasets, please use the `download_text_data.sh` script in the train folder.
This will download two datasets:
1. [Emotions dataset from huggingface](https://huggingface.co/datasets/emotion)
2. [GoEmotions dataset from google research](https://github.com/google-research/google-research/tree/master/goemotions)

After downloading, the datasets will be combined and stored in the text subfolder.

### Image Data
To download the image datasets, please use the `download_image_data.sh` script in the train folder.
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

Additional datasets that you can download yourself and add to the folder are available.
I recommend the following datasets:
3. [Jaffe Dataset](https://zenodo.org/record/3451524#.YlRNsTzb1H4)
   1. You need to download this dataset manually. Go to the link above and request access to the data.
   2. Download the data and put the zip file in the `data/train/images` folder.
   3. After that, run the `download_image_data.sh` script.
4. [CK+ Dataset](https://paperswithcode.com/dataset/ck)
   1. This data needs to be downloaded manually. Go to the link above and request the data.
5. [AffectNet Dataset](http://mohammadmahoor.com/affectnet/)
   1. You need to request access to the data yourself from the page above.
   2. Own labelling is recommended! Many default labels are incorrect.
6. [FFQH Dataset](https://github.com/NVlabs/ffhq-dataset)
   1. Download the data and label it manually. No labels are available.
7. [BU-3DFE Dataset](https://www.cs.binghamton.edu/~lijun/Research/3DFE/3DFE_Analysis.html)
   1. You need to request the data online. It is already labelled.

All the datasets above can be automatically extracted by the `download_image_data.sh` script.
Please take a look at the script to see how data should be formatted.

### Speech Data
To download the speech datasets, please use the `download_speech_data.sh` script in the train folder.
This will download these datasets:
1. [RAVDESS database](https://smartlaboratory.org/ravdess/)
2. [MELD dataset](https://affective-meld.github.io/)
3. [Crema D Dataset](https://www.tensorflow.org/datasets/catalog/crema_d):
   1. This dataset is stored in your tfds download folder (usually `/home/$USER/tensorflow-datasets`)

## Evaluation data
This part of the data has been collected by myself during my time at MIT.  
It can not be disclosed publicly because of privacy reasons of the subjects of the data stud

### Plant Data
The plant data should be cut to the experiment duration and then placed in the `plant` subfolder.  

