"""This script combines all the image datasets into one larger dataset"""

import glob
import os
import random
import shutil
import warnings

import numpy as np
from PIL import Image


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
            img.convert("RGB")
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
            img.convert("RGB")
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
            img.convert("RGB")
            img.save(
                os.path.join(
                    "data/train/image/test",
                    emotion,
                    os.path.basename(im)[:-4] + "png",
                )
            )
    print("JAFFE copying successful.")


if __name__ == "__main__":
    prepare_folders()

    copy_kaggle_dataset()
    copy_jaffe_dataset()
