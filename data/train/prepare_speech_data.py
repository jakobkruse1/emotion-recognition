"""This script combines all the speech datasets into one larger dataset"""

import glob
import os
import random
import shutil
import warnings


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


if __name__ == "__main__":
    prepare_folders()

    copy_ravdess_dataset()
