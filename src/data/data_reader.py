from abc import ABC, abstractmethod
from enum import Enum

import numpy as np
import tensorflow as tf


class Set(Enum):
    """
    Define the different set types that are available
    """

    TRAIN = 0
    VAL = 1
    TEST = 2


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

    @abstractmethod
    def get_data(
        self, which_set: Set, batch_size: int = 64
    ) -> tf.data.Dataset:
        """
        Main method which loads the data from disk into a Dataset instance

        :param which_set: Which set to use, can be either train, val or test
        :param batch_size: The batch size for the requested dataset
        :return: The Dataset instance to use in the emotion classifiers
        """
        raise NotImplementedError()  # pragma: no cover

    @abstractmethod
    def get_three_emotion_data(
        self, which_set: Set, batch_size: int = 64
    ) -> tf.data.Dataset:
        """
        Method that loads the dataset from disk and stores the labels
        in the ThreeEmotionSet instead of the NeutralEkmanEmotionSet
        :param which_set: train, val or test set distinguisher
        :param batch_size: the batch size for the dataset
        :return: The Dataset that contains data and labels
        """
        raise NotImplementedError()  # pragma: no cover

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
