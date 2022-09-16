"""This file implements that basic functions for data reading"""

from abc import ABC, abstractmethod
from enum import IntEnum
from typing import Dict, Tuple

import numpy as np
import tensorflow as tf


class Set(IntEnum):
    """
    Define the different set types that are available
    """

    TRAIN = 0
    VAL = 1
    TEST = 2
    ALL = 3


class DataReader(ABC):
    """
    The DataReader class is responsible for creating a tensorflow DataSet
    which is used for training and evaluating the emotion detection models.
    """

    def __init__(self, name: str, folder: str):
        """
        Initialize the data reader instance

        :param name: The name of the data reader instance
        :param folder: The folder that contains the data
        """
        self.name = name
        self.folder = folder
        self.num_batch = {Set.TRAIN: -1, Set.VAL: -1, Set.TEST: -1}

    @abstractmethod
    def get_seven_emotion_data(
        self, which_set: Set, batch_size: int = 64, parameters: Dict = None
    ) -> tf.data.Dataset:
        """
        Main method which loads the data from disk into a Dataset instance

        :param which_set: Which set to use, can be either train, val or test
        :param batch_size: The batch size for the requested dataset
        :param parameters: Additional parameters
        :return: The Dataset instance to use in the emotion classifiers
        """
        raise NotImplementedError()  # pragma: no cover

    @abstractmethod
    def get_three_emotion_data(
        self, which_set: Set, batch_size: int = 64, parameters: Dict = None
    ) -> tf.data.Dataset:
        """
        Method that loads the dataset from disk and stores the labels
        in the ThreeEmotionSet instead of the NeutralEkmanEmotionSet
        :param which_set: train, val or test set distinguisher
        :param batch_size: the batch size for the dataset
        :param parameters: Additional arguments
        :return: The Dataset that contains data and labels
        """
        raise NotImplementedError()  # pragma: no cover

    def get_emotion_data(
        self,
        emotions: str = "neutral_ekman",
        which_set: Set = Set.TRAIN,
        batch_size: int = 64,
        parameters: Dict = None,
    ) -> tf.data.Dataset:
        """
        Method that returns a dataset depending on the emotion set.

        :param emotions: The emotion set to use: neutral_ekman or three
        :param which_set: train, test or val set
        :param batch_size: The batch size for the dataset
        :param parameters: Additional arguments
        :return: The obtained dataset
        """
        if emotions == "neutral_ekman":
            return self.get_seven_emotion_data(
                which_set, batch_size, parameters
            )
        elif emotions == "three":
            return self.get_three_emotion_data(
                which_set, batch_size, parameters
            )
        else:
            raise ValueError(f'The emotion set "{emotions}" does not exist!')

    @staticmethod
    def convert_to_three_emotions(labels: np.ndarray) -> np.ndarray:
        """
        Convert the NeutralEkmanEmotion labels to the ThreeEmotionSet

        :param labels: The integer labels from 0-6 in NeutralEkman format
        :return: The integer labels from 0-2 in ThreeEmotion format
        """
        new_labels = labels.copy()
        conversion_dict = {0: 2, 1: 0, 2: 2, 3: 0, 4: 2, 5: 2, 6: 1}
        for old_val, new_val in conversion_dict.items():
            new_labels[labels == old_val] = new_val

        return new_labels

    @staticmethod
    def convert_to_three_emotions_onehot(labels: np.ndarray) -> np.ndarray:
        """
        Convert the NeutralEkmanEmotion labels to the ThreeEmotionSet

        :param labels: The integer labels from 0-6 in a one-hot encoding
            -> shape (n, 7)
        :return: The integer labels from 0-2 in ThreeEmotion format in
            one-hot encoding: shape (n,3)
        """
        assert labels.shape[1] == 7
        new_labels = np.zeros((labels.shape[0], 3))
        conversion_dict = {0: 2, 1: 0, 2: 2, 3: 0, 4: 2, 5: 2, 6: 1}
        for old_val, new_val in conversion_dict.items():
            new_labels[:, new_val] += labels[:, old_val]

        return new_labels

    @abstractmethod
    def get_labels(
        self, which_set: Set = Set.TRAIN, parameters: Dict = None
    ) -> np.ndarray:
        """
        Method that gets only the labels for the dataset that is specified

        :param which_set: Which set to use, train, val or test
        :param parameters: Parameter dictionary
        :return: An array of labels in shape (num_samples,)
        """
        raise NotImplementedError("Abstract method")  # pragma: no cover

    def cleanup(self, parameters: Dict = None) -> None:
        """
        Optional cleanup method that deletes unneccessary memory elements.

        :param parameters: Parameters that might be required
        """
        pass

    @staticmethod
    def convert_to_numpy(
        dataset: tf.data.Dataset,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Converts a given tensorflow dataset into a single numpy array

        :param dataset: The dataset to convert to numpy
        :return: Tuple containing two array:
            - numpy array containing data from all batches
            - numpy array containing labels from all batches
        """
        np_data = []
        np_labels = []
        for data, labels in dataset:
            np_data.append(data.numpy())
            np_labels.append(labels.numpy())
        return np.concatenate(np_data, axis=0), np.concatenate(
            np_labels, axis=0
        )

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
