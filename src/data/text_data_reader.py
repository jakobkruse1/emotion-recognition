"""This file implements the data reading functionality for text data."""

import os

import pandas as pd
import tensorflow as tf

from src.data.data_reader import DataReader, Set


class TextDataReader(DataReader):
    """
    Class that reads the CSV datasets from the data/train/text folder
    """

    def __init__(self):
        """
        Initialization for the class
        """
        super().__init__("text", "data/train/text")
        self.file_map = {
            Set.TRAIN: "final_train.csv",
            Set.VAL: "final_val.csv",
            Set.TEST: "final_test.csv",
        }

    def get_data(
        self, which_set: Set, batch_size: int = 64
    ) -> tf.data.Dataset:
        """
        Main data reading function which reads the CSV file into a dataset

        :param which_set: Which dataset to use - train, val or test
        :param batch_size: The batch size for the resulting dataset
        :return: The tensorflow Dataset instance
        """
        csv_file_path = os.path.join(self.folder, self.file_map[which_set])
        data = pd.read_csv(
            csv_file_path, sep="\t", header=None, names=["text", "label"]
        )
        labels = tf.keras.utils.to_categorical(data.pop("label"))
        dataset = tf.data.Dataset.from_tensor_slices((data, labels))
        dataset = (
            dataset.shuffle(1024)
            .cache()
            .batch(batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )
        return dataset

    def get_three_emotion_data(
        self, which_set: Set, batch_size: int = 64
    ) -> tf.data.Dataset:
        """
        Main data reading function which reads the CSV file into a dataset
        and also converts the emotion labels to the three emotion space.

        :param which_set: Which dataset to use - train, val or test
        :param batch_size: The batch size for the resulting dataset
        :return: The tensorflow Dataset instance
        """
        csv_file_path = os.path.join(self.folder, self.file_map[which_set])
        data = pd.read_csv(
            csv_file_path, sep="\t", header=None, names=["text", "label"]
        )
        labels = data.pop("label")
        labels = self.convert_to_three_emotions(labels)
        labels = tf.keras.utils.to_categorical(labels)
        dataset = tf.data.Dataset.from_tensor_slices((data, labels))
        dataset = (
            dataset.shuffle(1024)
            .cache()
            .batch(batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )
        return dataset
