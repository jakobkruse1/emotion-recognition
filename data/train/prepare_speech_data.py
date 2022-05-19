"""This script combines all the speech datasets into one larger dataset"""

import glob
import os
import random
import shutil
import warnings

import numpy as np
import pandas as pd
import tensorflow_datasets as tfds


def prepare_folders():
    folders = [
        "data/train/speech/train",
        "data/train/speech/val",
        "data/train/speech/test",
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
        if not os.path.exists(folder):
            os.makedirs(folder)
            for subfolder in subfolders:
                os.makedirs(os.path.join(folder, subfolder))


def copy_ravdess_dataset():
    if not os.path.exists("data/train/speech/ravdess"):
        warnings.warn("Ravdess Dataset not downloaded. Skipping!")
        return
    print("Copying RAVDESS dataset.")
    emotions = [
        "angry",
        "disgust",
        "fear",
        "happy",
        "neutral",
        "sad",
        "surprise",
    ]
    emotion_map = {
        "05": "angry",
        "07": "disgust",
        "06": "fear",
        "03": "happy",
        "01": "neutral",
        "02": "neutral",
        "04": "sad",
        "08": "surprise",
    }
    emotion_files = {val: [] for val in emotions}
    for speech_path in glob.glob("data/train/speech/ravdess/**/*.wav"):
        file_name = speech_path.split("/")[-1]
        emotion = emotion_map[file_name[6:8]]
        emotion_files[emotion].append(speech_path)

    for emotion, file_list in emotion_files.items():
        random.Random(42).shuffle(file_list)
        for file in file_list[: int(0.6 * len(file_list))]:
            shutil.copyfile(
                file,
                os.path.join(
                    f"data/train/speech/train/{emotion}/{file.split('/')[-1]}"
                ),
            )
        for file in file_list[
            int(0.6 * len(file_list)) : int(0.8 * len(file_list))
        ]:
            shutil.copyfile(
                file,
                os.path.join(
                    f"data/train/speech/val/{emotion}/{file.split('/')[-1]}"
                ),
            )
        for file in file_list[int(0.8 * len(file_list)) :]:
            shutil.copyfile(
                file,
                os.path.join(
                    f"data/train/speech/test/{emotion}/{file.split('/')[-1]}"
                ),
            )


def copy_meld_dataset():
    if not os.path.exists("data/train/speech/meld"):
        warnings.warn("MELD Dataset not downloaded. Skipping!")
        return
    print("Copying MELD dataset.")
    for file, which_set in [
        ("train", "train"),
        ("dev", "val"),
        ("test", "test"),
    ]:
        csv_file = f"data/train/speech/meld/{file}_sent_emo.csv"
        metadata = pd.read_csv(csv_file, delimiter=",")
        emotions = [
            "anger",
            "disgust",
            "fear",
            "joy",
            "neutral",
            "sadness",
            "surprise",
        ]
        folders = [
            "angry",
            "disgust",
            "fear",
            "happy",
            "neutral",
            "sad",
            "surprise",
        ]
        files = glob.glob(f"data/train/speech/meld/{which_set}/*.wav")
        dialogues = np.array(
            [int(file.split("dia")[-1].split("_")[0]) for file in files]
        )
        utterances = np.array(
            [int(file.split("utt")[-1].split(".")[0]) for file in files]
        )
        for index, row in metadata.iterrows():
            emotion = row["Emotion"]
            folder = folders[emotions.index(emotion)]
            file_index = np.where(
                np.logical_and(
                    dialogues == row["Dialogue_ID"],
                    utterances == row["Utterance_ID"],
                )
            )[0]
            if len(file_index) == 0:
                continue
            elif len(file_index) == 2:
                to_use = 0 if "final" in files[file_index[0]] else 1
                file_index = file_index[to_use]
            else:
                file_index = file_index[0]
            shutil.copyfile(
                files[file_index],
                os.path.join(
                    "data/train/speech",
                    which_set,
                    folder,
                    os.path.basename(files[file_index]),
                ),
            )


def download_crema_d_dataset():
    train, dev, test = tfds.load(
        "crema_d", split=["train", "validation", "test"], shuffle_files=False
    )


if __name__ == "__main__":
    prepare_folders()

    copy_ravdess_dataset()
    copy_meld_dataset()

    download_crema_d_dataset()
