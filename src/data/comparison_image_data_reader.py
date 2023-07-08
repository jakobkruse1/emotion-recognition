""" This file implements the data reading functionality for the image data
from the comparison dataset. """

import os
from typing import Dict

import numpy as np
import tensorflow as tf

from src.data.data_reader import DataReader, Set
from src.utils import reader_main


class ComparisonImageDataReader(DataReader):
    """
    Class that reads the comparison dataset image data
    """

    def __init__(self, name: str = "comparison_image", folder: str = None):
        """
        Initialization for the class

        :param name: The name for the image reader.
        :param folder: The folder that contains the data.
        """
        super().__init__(
            name, folder or os.path.join("data", "comparison_dataset", "image")
        )

    def get_seven_emotion_data(
        self, which_set: Set, batch_size: int = 64, parameters: Dict = None
    ) -> tf.data.Dataset:
        """
        Main data reading function which reads the images into a dataset

        :param which_set: Which dataset to use - only test is allowed here
        :param batch_size: The batch size for the resulting dataset
        :param parameters: Additional parameters
        :return: The tensorflow Dataset instance
        """
        assert (
            which_set == Set.TEST
        ), "The comparison dataset can only be used for testing."
        parameters = parameters or {}
        dataset = tf.keras.utils.image_dataset_from_directory(
            self.folder,
            shuffle=False,
            batch_size=batch_size,
            image_size=(224, 224),
            label_mode="categorical",
            color_mode="grayscale",
            class_names=[
                "angry",
                "surprise",
                "disgust",
                "happy",
                "fear",
                "sad",
                "neutral",
            ],
        )
        self.num_batch[which_set] = int(dataset.cardinality().numpy())
        fraction = parameters.get("fraction", 0.652)
        dataset = dataset.map(
            lambda x, y: (
                tf.image.resize(tf.image.central_crop(x, fraction), (48, 48)),
                y,
            )
        )
        return dataset

    def get_three_emotion_data(
        self, which_set: Set, batch_size: int = 64, parameters: Dict = None
    ) -> tf.data.Dataset:
        """
        Main data reading function which reads the image folders into a dataset
        and also converts the emotion labels to the three emotion space.

        :param which_set: Which dataset to use - test only
        :param batch_size: The batch size for the resulting dataset
        :param parameters: Additional arguments
        :return: The tensorflow Dataset instance
        """
        assert (
            which_set == Set.TEST
        ), "The comparison dataset can only be used for testing."
        parameters = parameters or {}
        dataset = tf.keras.utils.image_dataset_from_directory(
            self.folder,
            shuffle=False,
            batch_size=batch_size,
            image_size=(224, 224),
            label_mode="categorical",
            color_mode="grayscale",
            class_names=[
                "angry",
                "surprise",
                "disgust",
                "happy",
                "fear",
                "sad",
                "neutral",
            ],
        )
        self.num_batch[which_set] = int(dataset.cardinality().numpy())
        fraction = parameters.get("fraction", 0.652)
        dataset = dataset.map(
            lambda x, y: (
                tf.image.resize(tf.image.central_crop(x, fraction), (48, 48)),
                y,
            )
        )
        dataset = dataset.map(
            lambda x, y: tf.numpy_function(
                func=self.map_emotions,
                inp=[x, y],
                Tout=(tf.float32, tf.float32),
            )
        )
        return dataset

    def get_labels(
        self, which_set: Set = Set.TRAIN, parameters: Dict = None
    ) -> np.ndarray:
        """
        Get the labels for the image dataset in an array

        :param which_set: Train, val or test set - only test allowed here
        :param parameters: Parameter dictionary
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
    reader = ComparisonImageDataReader()
    reader_main(reader, {})


if __name__ == "__main__":  # pragma: no cover
    _main()
