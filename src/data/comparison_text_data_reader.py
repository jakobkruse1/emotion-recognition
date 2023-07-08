"""This file implements the data reading functionality for text data
from the comparison dataset."""

import os
from typing import Dict

import numpy as np
import pandas as pd
import tensorflow as tf

from src.data.data_reader import DataReader, Set
from src.utils import reader_main


class ComparisonTextDataReader(DataReader):
    """
    Class that reads the CSV datasets from the data/train/text folder
    """

    def __init__(self, folder: str = None):
        """
        Initialization for the class

        :param folder: The folder that contains the data.
        """
        super().__init__(
            "comparison_text",
            folder or os.path.join("data", "comparison_dataset", "text"),
        )
        self.emotion_labels = {
            "angry": 0,
            "surprise": 1,
            "disgust": 2,
            "happy": 3,
            "fear": 4,
            "sad": 5,
            "neutral": 6,
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
        assert (
            which_set == Set.TEST
        ), "The comparison dataset can only be used for testing."
        all_text = np.empty((0,))
        all_labels = np.empty((0,))
        for emotion, label in self.emotion_labels.items():
            csv_file_path = os.path.join(self.folder, f"{emotion}.csv")
            data = pd.read_csv(
                csv_file_path, sep="|", header=None, names=["text"]
            )
            all_labels = np.concatenate(
                [all_labels, label * np.ones((len(data),))]
            )
            text = data.pop("text")
            text[text.isna()] = ""
            all_text = np.concatenate([all_text, text], axis=0)
        all_labels = tf.keras.utils.to_categorical(all_labels)
        dataset = tf.data.Dataset.from_tensor_slices((all_text, all_labels))
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
        seven_dataset = self.get_seven_emotion_data(
            which_set, batch_size, parameters
        )
        three_dataset = seven_dataset.map(
            lambda x, y: tf.numpy_function(
                func=self.map_emotions,
                inp=[x, y],
                Tout=(tf.string, tf.float32),
            )
        )

        return three_dataset

    def get_labels(
        self, which_set: Set = Set.TRAIN, parameters: Dict = None
    ) -> np.ndarray:
        """
        Get the labels for the text dataset that is specified in an array

        :param which_set: Train, val or test set
        :param parameters: Parameter dict (unused)
        :return: The labels in an array of shape (num_samples,)
        """
        parameters = parameters or {}
        dataset = self.get_seven_emotion_data(which_set, parameters=parameters)
        all_labels = np.empty((0,))
        for images, labels in dataset:
            all_labels = np.concatenate(
                [all_labels, np.argmax(labels.numpy(), axis=1)], axis=0
            )

        return all_labels


def _main():  # pragma: no cover
    reader = ComparisonTextDataReader()
    reader_main(reader, {})


if __name__ == "__main__":  # pragma: no cover
    _main()
