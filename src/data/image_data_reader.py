"""This file implements the data reading functionality for image data."""

import os

import numpy as np
import tensorflow as tf

from src.data.data_reader import DataReader, Set


class ImageDataReader(DataReader):
    """
    Class that reads the CSV datasets from the data/train/text folder
    """

    def __init__(self, folder: str = "data/train/image"):
        """
        Initialization for the class
        """
        super().__init__("image", folder)
        self.folder_map = {
            Set.TRAIN: "train",
            Set.VAL: "val",
            Set.TEST: "test",
        }

    def get_seven_emotion_data(
        self, which_set: Set, batch_size: int = 64, **kwargs
    ) -> tf.data.Dataset:
        """
        Main data reading function which reads the images into a dataset

        :param which_set: Which dataset to use - train, val or test
        :param batch_size: The batch size for the resulting dataset
        :param kwargs: Additional parameters
        :return: The tensorflow Dataset instance
        """
        shuffle = kwargs.get(
            "shuffle", True if which_set == Set.TRAIN else False
        )
        dataset = tf.keras.utils.image_dataset_from_directory(
            os.path.join(self.folder, self.folder_map[which_set]),
            shuffle=shuffle,
            batch_size=batch_size,
            image_size=(48, 48),
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
        return dataset

    def get_three_emotion_data(
        self, which_set: Set, batch_size: int = 64, **kwargs
    ) -> tf.data.Dataset:
        """
        Main data reading function which reads the CSV file into a dataset
        and also converts the emotion labels to the three emotion space.

        :param which_set: Which dataset to use - train, val or test
        :param batch_size: The batch size for the resulting dataset
        :param kwargs: Additional arguments
        :return: The tensorflow Dataset instance
        """
        shuffle = kwargs.get(
            "shuffle", True if which_set == Set.TRAIN else False
        )
        dataset = tf.keras.utils.image_dataset_from_directory(
            os.path.join(self.folder, self.folder_map[which_set]),
            shuffle=shuffle,
            batch_size=batch_size,
            image_size=(48, 48),
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
        dataset = dataset.map(
            lambda x, y: tf.numpy_function(
                func=self.map_emotions,
                inp=[x, y],
                Tout=(tf.float32, tf.float32),
            )
        )

        return dataset

    @staticmethod
    def map_emotions(data, labels):
        """
        Conversion function that is applied when three emotion labels are
        required.
        """
        new_labels = DataReader.convert_to_three_emotions_onehot(
            labels
        ).astype(np.float32)
        return data, new_labels

    def get_labels(self, which_set: Set = Set.TRAIN) -> np.ndarray:
        """
        Get the labels for the text dataset that is specified in an array

        :param which_set: Train, val or test set
        :return: The labels in an array of shape (num_samples,)
        """
        dataset = self.get_seven_emotion_data(which_set, shuffle=False)
        all_labels = np.empty((0,))
        for images, labels in dataset:
            all_labels = np.concatenate(
                [all_labels, np.argmax(labels.numpy(), axis=1)], axis=0
            )

        return all_labels
