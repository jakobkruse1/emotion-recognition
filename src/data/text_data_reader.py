"""This file implements the data reading functionality for text data."""

import os
from typing import Dict

import numpy as np
import pandas as pd
import tensorflow as tf

from src.data.data_reader import DataReader, Set
from src.utils import reader_main


class TextDataReader(DataReader):
    """
    Class that reads the CSV datasets from the data/train/text folder
    """

    def __init__(self, folder: str = "data/train/text"):
        """
        Initialization for the class

        :param folder: The folder that contains the data.
        """
        super().__init__("text", folder or "data/train/text")
        self.file_map = {
            Set.TRAIN: "final_train.csv",
            Set.VAL: "final_val.csv",
            Set.TEST: "final_test.csv",
        }

    def get_seven_emotion_data(
        self, which_set: Set, batch_size: int = 64, parameters: Dict = None
    ) -> tf.data.Dataset:
        """
        Main data reading function which reads the CSV file into a dataset

        :param which_set: Which dataset to use - train, val or test
        :param batch_size: The batch size for the resulting dataset
        :param parameters: Additional parameters
        :return: The tensorflow Dataset instance
        """
        parameters = parameters or {}
        shuffle = parameters.get(
            "shuffle", True if which_set == Set.TRAIN else False
        )
        csv_file_path = os.path.join(self.folder, self.file_map[which_set])
        data = pd.read_csv(
            csv_file_path, sep="\t", header=None, names=["text", "label"]
        )
        labels = tf.keras.utils.to_categorical(data.pop("label"))
        dataset = tf.data.Dataset.from_tensor_slices((data, labels))
        if shuffle:
            dataset = dataset.shuffle(1024)
        dataset = dataset.cache().batch(batch_size).prefetch(tf.data.AUTOTUNE)
        self.num_batch[which_set] = int(dataset.cardinality().numpy())
        return dataset

    def get_three_emotion_data(
        self, which_set: Set, batch_size: int = 64, parameters: Dict = None
    ) -> tf.data.Dataset:
        """
        Main data reading function which reads the CSV file into a dataset
        and also converts the emotion labels to the three emotion space.

        :param which_set: Which dataset to use - train, val or test
        :param batch_size: The batch size for the resulting dataset
        :param parameters: Additional arguments
        :return: The tensorflow Dataset instance
        """
        parameters = parameters or {}
        shuffle = parameters.get(
            "shuffle", True if which_set == Set.TRAIN else False
        )
        csv_file_path = os.path.join(self.folder, self.file_map[which_set])
        data = pd.read_csv(
            csv_file_path, sep="\t", header=None, names=["text", "label"]
        )
        labels = data.pop("label")
        labels = self.convert_to_three_emotions(labels)
        labels = tf.keras.utils.to_categorical(labels)
        dataset = tf.data.Dataset.from_tensor_slices((data, labels))
        if shuffle:
            dataset = dataset.shuffle(1024)
        dataset = dataset.cache().batch(batch_size).prefetch(tf.data.AUTOTUNE)
        self.num_batch[which_set] = int(dataset.cardinality().numpy())
        return dataset

    def get_labels(
        self, which_set: Set = Set.TRAIN, parameters: Dict = None
    ) -> np.ndarray:
        """
        Get the labels for the text dataset that is specified in an array

        :param which_set: Train, val or test set
        :param parameters: Parameter dict (unused)
        :return: The labels in an array of shape (num_samples,)
        """
        csv_file_path = os.path.join(self.folder, self.file_map[which_set])
        labels = pd.read_csv(
            csv_file_path, delimiter="\t", usecols=[1], header=None
        )
        return np.reshape(labels.to_numpy(), (-1,))


def _main():  # pragma: no cover
    reader = TextDataReader()
    reader_main(reader, {})


if __name__ == "__main__":  # pragma: no cover
    _main()
