""" This file contains a base class for data readers that read experiment
related data and implements common functionality. """

from abc import abstractmethod
from typing import Dict, List

import numpy as np
import tensorflow as tf

from src.data.data_reader import DataReader, Set


class ExperimentDataReader(DataReader):
    """
    This is the base class for all experiment related data readers.
    """

    def __init__(self, name: str, folder: str) -> None:
        """
        This function initializes the abstract data reader for experiment data.

        :param name: The name of the data reader to create.
        :param folder: The folder where the data is located.
        """
        super().__init__(name, folder)
        self.emotions = [
            "neutral",
            "joy",
            "disgust",
            "anger",
            "surprise",
            "sadness",
            "fear",
        ]
        self.emotion_labels = {
            "anger": 0,
            "surprise": 1,
            "disgust": 2,
            "joy": 3,
            "fear": 4,
            "sadness": 5,
            "neutral": 6,
        }
        self.emotion_times = self.get_emotion_times()

    def get_emotion_times(self) -> Dict[str, Dict[str, float]]:
        """
        This function returns start and end times for every emotion in the
        experiments.

        :return: The start and end time for every emotion.
        """
        starts = [0]
        ends = []
        sent_times = [41.03, 98.23, 147.70, 228.16, 287.30, 406.20, 601.80]
        for i in range(6):
            starts.append(sent_times[i] + 12)
            ends.append(sent_times[i] + 12)
        ends.append(sent_times[6] + 12)
        emotion_times = {}
        for emotion, start, end in zip(self.emotions, starts, ends):
            emotion_times[emotion] = {"start": start, "end": end}
        return emotion_times

    @abstractmethod
    def get_seven_emotion_data(
        self, which_set: Set, batch_size: int = 64, parameters: Dict = None
    ) -> tf.data.Dataset:
        """
        The abstract method for getting the dataset to train on.

        :param which_set: Training, Validation or Test Set
        :param batch_size: Batch Size for the dataset
        :param parameters: Additional parameters.
        :return: A tensorflow Dataset instance.
        """
        raise NotImplementedError()  # pragma: no cover

    @abstractmethod
    def get_three_emotion_data(
        self, which_set: Set, batch_size: int = 64, parameters: Dict = None
    ) -> tf.data.Dataset:
        """
        The abstract method for getting the dataset to train on.
        This method should return only three emotions.

        :param which_set: Training, Validation or Test Set
        :param batch_size: Batch Size for the dataset
        :param parameters: Additional parameters.
        :return: A tensorflow Dataset instance.
        """
        raise NotImplementedError()  # pragma: no cover

    @abstractmethod
    def get_labels(
        self, which_set: Set = Set.TRAIN, parameters: Dict = None
    ) -> np.ndarray:
        """
        Return the labels for the unsorted data in the dataset.

        :param which_set: Which set to get labels for
        :param parameters: Additional parameters
        :return: Numpy array of labels.
        """
        raise NotImplementedError()  # pragma: no cover

    @staticmethod
    def get_complete_data_indices() -> List[int]:
        """
        Static method that returns all experiment indices that have complete
        data and are supposed to be used in the evaluation.

        :return: List of experiment indices.
        """
        complete_data = (
            list(range(5, 9))
            + list(range(11, 15))
            + [16]
            + list(range(18, 21))
            + list(range(24, 58))
            + list(range(59, 62))
            + list(range(63, 66))
            + list(range(68, 70))
        )

        return complete_data
