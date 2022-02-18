"""This script combines all the text datasets into one larger dataset"""

import csv
import os
from typing import Tuple

import numpy as np

from src.emotion_set import EkmanNeutralEmotions, EmotionMapper


def read_txt_file(file_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reads a txt file which contains data points from the huggingface dataset

    :param file_path: The path to the txt file to read
    :return: Tuple with all texts and labels from the file
    """
    emotion_set = EkmanNeutralEmotions()
    emotion_mapper = EmotionMapper()
    emotion_names = list(emotion_set.emotion_names)
    texts = []
    labels = []
    base_path = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(base_path, file_path), "r") as file:
        for line in file:
            text, emotion = line.strip().split(";")
            ekman_emotion = emotion_mapper.map_emotion(emotion)
            index = emotion_names.index(ekman_emotion)
            texts.append(text)
            labels.append(index)
    return np.asarray(texts), np.asarray(labels)


def read_tsv_file(file_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    This function reads a tsv file from the goemotions dataset.

    :param file_path: The path to the tsv file
    :return: Tuple with all texts and labels from the file
    """
    base_path = os.path.dirname(os.path.realpath(__file__))
    emotion_set = EkmanNeutralEmotions()
    emotion_mapper = EmotionMapper()
    final_emotion_names = list(emotion_set.emotion_names)

    emotions = []
    with open(os.path.join(base_path, "text/emotions.txt"), "r") as em_file:
        for line in em_file:
            emotions.append(line.strip())
    texts = []
    labels = []
    with open(os.path.join(base_path, file_path), "r") as tsv_file:
        reader = csv.reader(tsv_file, delimiter="\t")
        for line in reader:
            text = line[0]
            emotion_ids = line[1].split(",")
            emotion_names = [emotions[int(index)] for index in emotion_ids]
            ekman_emotions = [
                emotion_mapper.map_emotion(em) for em in emotion_names
            ]
            if ekman_emotions.count(ekman_emotions[0]) == len(ekman_emotions):
                # All emotions in the dataset correspond to the same ekman em.
                texts.append(text)
                labels.append(final_emotion_names.index(ekman_emotions[0]))

    return np.asarray(texts), np.asarray(labels)


if __name__ == "__main__":
    # First, read the huggingface emotions dataset
    train_texts, train_labels = read_txt_file("text/train.txt")
    val_texts, val_labels = read_txt_file("text/validation.txt")
    test_texts, test_labels = read_txt_file("text/test.txt")
    # Secondly, read the GoEmotions dataset files
    train_texts2, train_labels2 = read_tsv_file("text/train.tsv")
    val_texts2, val_labels2 = read_tsv_file("text/dev.tsv")
    test_texts2, test_labels2 = read_tsv_file("text/test.tsv")

    # Combine the datasets
    train_texts = np.concatenate([train_texts, train_texts2])
    val_texts = np.concatenate([val_texts, val_texts2])
    test_texts = np.concatenate([test_texts, test_texts2])
    train_labels = np.concatenate([train_labels, train_labels2])
    val_labels = np.concatenate([val_labels, val_labels2])
    test_labels = np.concatenate([test_labels, test_labels2])

    # Print some details about the data
    def print_label_distribution(labels, name):
        emotion_set = EkmanNeutralEmotions()
        emotion_names = list(emotion_set.emotion_names)
        labels, counts = np.unique(labels, return_counts=True)
        labels = [emotion_names[label] for label in labels]
        print(f"{name}: {dict(zip(labels, counts))}")

    print("Label distribution in the datasets:")
    print_label_distribution(train_labels, "Train")
    print_label_distribution(val_labels, "Val")
    print_label_distribution(test_labels, "Test")

    # Store in combined csv files
    # TODO
