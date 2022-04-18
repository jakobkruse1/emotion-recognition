"""This script combines all the image datasets into one larger dataset"""

import glob
import os
import random
import shutil
import warnings

import numpy as np
import pandas as pd
from PIL import Image, ImageOps


def prepare_folders():
    """
    Simple function creating all folders required for the dataset
    """
    folders = [
        "data/train/image/train",
        "data/train/image/val",
        "data/train/image/test",
    ]
    subfolders = [
        "angry",
        "disgust",
        "fear",
        "happy",
        "neutral",
        "sad",
        "surprise",
    ]
    for folder in folders:
        os.makedirs(folder)
        for subfolder in subfolders:
            os.makedirs(os.path.join(folder, subfolder))


def copy_kaggle_dataset(train_split=0.8):
    """
    This function copies the images from the kaggle dataset over.
    :param train_split: Float between 0 and 1 determining how big training
        dataset is relatively.
    """
    if not os.path.exists("data/train/image/images"):
        warnings.warn("Kaggle Dataset not downloaded. Skipping!")
        return
    print("Copying kaggle dataset.")
    shutil.copytree(
        "data/train/image/images/validation",
        "data/train/image/test",
        dirs_exist_ok=True,
    )
    subfolders = [
        "angry",
        "disgust",
        "fear",
        "happy",
        "neutral",
        "sad",
        "surprise",
    ]
    for subfolder in subfolders:
        workdir = os.path.join("data/train/image/images/train", subfolder)
        file_list = np.array(os.listdir(workdir))
        np.random.shuffle(file_list)
        for file in file_list[: int(train_split * file_list.shape[0])]:
            shutil.copyfile(
                os.path.join(workdir, file),
                os.path.join("data/train/image/train", subfolder, file),
            )
        for file in file_list[int(train_split * file_list.shape[0]) :]:
            shutil.copyfile(
                os.path.join(workdir, file),
                os.path.join("data/train/image/val", subfolder, file),
            )
    print("Kaggle copying successful.")


def copy_jaffe_dataset():
    """
    This function copies the images from the kaggle dataset over.
    :param train_split: Float between 0 and 1 determining how big training
        dataset is relatively.
    """
    if not os.path.exists("data/train/image/jaffedbase"):
        warnings.warn("JAFFE Dataset not downloaded. Skipping!")
        return
    print("Copying JAFFE dataset.")
    emotions = {
        "AN": "angry",
        "DI": "disgust",
        "FE": "fear",
        "HA": "happy",
        "NE": "neutral",
        "SA": "sad",
        "SU": "surprise",
    }
    images = {}
    for emotion in emotions.values():
        images[emotion] = []
    for image_path in glob.glob("data/train/image/jaffedbase/*.tiff"):
        images[emotions[os.path.basename(image_path)[3:5]]].append(image_path)
    for emotion, image_list in images.items():
        random.shuffle(image_list)
        for im in image_list[: int(0.6 * len(image_list))]:
            # Copy training
            img = Image.open(im)
            img = img.convert("RGB")
            img = img.resize((48, 48))
            img.save(
                os.path.join(
                    "data/train/image/train",
                    emotion,
                    os.path.basename(im)[:-4] + "png",
                )
            )
        for im in image_list[
            int(0.6 * len(image_list)) : int(0.8 * len(image_list))
        ]:
            # Copy val
            img = Image.open(im)
            img = img.convert("RGB")
            img = img.resize((48, 48))
            img.save(
                os.path.join(
                    "data/train/image/val",
                    emotion,
                    os.path.basename(im)[:-4] + "png",
                )
            )
        for im in image_list[int(0.8 * len(image_list)) :]:
            # Copy test
            img = Image.open(im)
            img = img.convert("RGB")
            img = img.resize((48, 48))
            img.save(
                os.path.join(
                    "data/train/image/test",
                    emotion,
                    os.path.basename(im)[:-4] + "png",
                )
            )
    print("JAFFE copying successful.")


def copy_fer_dataset(logging=False):
    """
    Function that prepares the FER2013 dataset for training classifiers
    :param logging: Activate loggin for correctness check of the data
    """
    if not os.path.exists("data/train/image/fer2013"):
        warnings.warn("FER2013 Dataset not downloaded. Skipping!")
        return
    print("Copying FER2013 dataset.")
    image_data = pd.read_csv(
        "data/train/image/fer2013/fer2013.csv",
        delimiter=",",
        header=0,
        usecols=[1],
    )
    label_data = pd.read_csv(
        "data/train/image/fer2013/fer_labels.csv", delimiter=",", header=0
    )
    folders = {"Training": "train", "PublicTest": "val", "PrivateTest": "test"}
    emotions = [
        "neutral",
        "happy",
        "surprise",
        "sad",
        "angry",
        "disgust",
        "fear",
    ]
    all_labels = label_data.to_numpy()[:, 2:12]
    for index in range(image_data.shape[0]):
        emotion_index = np.argmax(all_labels[index, :])
        intermed = all_labels[index, :].copy()
        intermed[emotion_index] = 0
        if emotion_index < 7 and all_labels[index, emotion_index] > np.max(
            intermed
        ):
            emotion = emotions[emotion_index]
            image = np.reshape(
                np.fromstring(image_data.iloc[index][0], sep=" "), (48, 48)
            )
            im = Image.fromarray(image)
            im = im.convert("RGB")
            save_path = os.path.join(
                "data/train/image",
                folders[label_data.iloc[index, 0]],
                emotion,
                f"fer_{index}.jpeg",
            )
            im.save(save_path)
        elif logging:
            print(f"Skipping index {index}, reason: {all_labels[index, :]}")
    print("FER2013 copying successful.")


def copy_affectnet_dataset():
    """
    Function that reads the affectnet dataset into the corresponding image
    folders and divides it into train, validation and test images.
    """
    if not os.path.exists("data/train/image/AffectNet"):
        warnings.warn("AffectNet Dataset not downloaded. Skipping!")
        return
    print("Copying AffectNet dataset.")
    image_data = pd.read_csv(
        "data/train/image/AffectNet/AffectNet_images_all.csv",
        delimiter=",",
        header=0,
        names=["labels", "images"],
    )
    labels = image_data.labels.to_numpy()
    emotions = [
        "angry",
        "fear",
        "happy",
        "sad",
        "surprise",
        "neutral",
        "contempt",
        "disgust",
    ]
    for emotion_index in [0, 1, 2, 3, 4, 5, 7]:
        emotion_indices = np.where(labels == emotion_index)[0]
        emotion = emotions[emotion_index]
        for count, image_index in enumerate(emotion_indices):
            image = np.reshape(
                np.fromstring(image_data.iloc[image_index][1], sep=" "),
                (100, 100),
            )
            im = Image.fromarray(image)
            im = im.convert("RGB")
            im = im.resize((48, 48))
            if count / emotion_indices.shape[0] <= 0.6:
                folder = "train"
            elif count / emotion_indices.shape[0] <= 0.8:
                folder = "val"
            else:
                folder = "test"
            save_path = os.path.join(
                "data/train/image",
                folder,
                emotion,
                f"aff_{image_index}.jpeg",
            )
            im.save(save_path)
    print("AffectNet copying successful.")


def copy_ffhq_dataset():
    """
    Function that reads the ffhq dataset into the corresponding image folders
    and divides it into train, validation and test images.
    """
    if not os.path.exists("data/train/image/FFHQ"):
        warnings.warn("FFHQ Dataset not downloaded. Skipping!")
        return
    print("Copying FFHQ dataset.")
    label_data = pd.read_csv(
        "data/train/image/FFHQ/FFHQ_6033.csv",
        delimiter=",",
        header=0,
        names=["image_id", "labels"],
    )
    labels = label_data.labels.to_numpy()
    emotions = [
        "angry",
        "fear",
        "happy",
        "sad",
        "surprise",
        "neutral",
        "contempt",
        "disgust",
    ]
    for emotion_index in [0, 1, 2, 3, 4, 5, 7]:
        emotion_indices = np.where(labels == emotion_index)[0]
        emotion = emotions[emotion_index]
        for count, image_index in enumerate(emotion_indices):
            image_name = label_data.iloc[image_index][0]
            if not os.path.exists(
                f"data/train/image/FFHQ/images/{image_name:05d}.png"
            ):
                # print(f"Image {image_name} missing")
                continue
            if count / emotion_indices.shape[0] <= 0.6:
                folder = "train"
            elif count / emotion_indices.shape[0] <= 0.8:
                folder = "val"
            else:
                folder = "test"
            im = Image.open(
                f"data/train/image/FFHQ/images/{image_name:05d}.png"
            )
            im = ImageOps.grayscale(im)
            im = im.resize((48, 48))
            save_path = os.path.join(
                "data/train/image",
                folder,
                emotion,
                f"ffhq_{image_name:05d}.jpeg",
            )
            im.save(save_path)
    print("FFHQ copying successful.")


def copy_ckplus_dataset():
    """
    Function that reads the CK+ dataset into the corresponding image folders
    and divides it into train, validation and test images.
    """
    if not os.path.exists("data/train/image/CK+"):
        warnings.warn("CK+ Dataset not downloaded. Skipping!")
        return
    print("Copying CK+ dataset.")
    emotions = [
        "neutral",
        "angry",
        "contempt",
        "disgust",
        "fear",
        "happy",
        "sad",
        "surprise",
    ]
    emotion_images = {emotion: [] for emotion in emotions}
    emotion_files = glob.glob(
        "data/train/image/CK+/Emotion/**/**/*_emotion.txt"
    )
    for emotion_file in emotion_files:
        with open(emotion_file, "r") as file:
            emotion = int(float(file.readline().strip()))
        if emotion == 2:
            continue
        image_path = emotion_file.replace("Emotion", "cohn-kanade-images")
        image_path = image_path.replace("_emotion.txt", ".png")

        im = Image.open(image_path)
        width, height = im.size
        new_size = min(width, height)
        left = (width - new_size) / 2
        top = (height - new_size) / 2
        right = (width + new_size) / 2
        bottom = (height + new_size) / 2
        im = im.crop((left, top, right, bottom))
        im = im.resize((48, 48))
        im.convert("RGB")
        emotion_images[emotions[emotion]].append(im)

    for emotion in emotions:
        for count, im in enumerate(emotion_images[emotion]):
            if count / len(emotion_images[emotion]) <= 0.6:
                folder = "train"
            elif count / len(emotion_images[emotion]) <= 0.8:
                folder = "val"
            else:
                folder = "test"
            save_path = os.path.join(
                "data/train/image",
                folder,
                emotion,
                f"ckplus_{count:05d}.jpeg",
            )
            im.save(save_path)

    print("CK+ copying successful.")


if __name__ == "__main__":
    prepare_folders()

    copy_kaggle_dataset()
    copy_jaffe_dataset()
    copy_fer_dataset()
    copy_affectnet_dataset()
    copy_ffhq_dataset()
    copy_ckplus_dataset()
