"""Implement an emotion classifier base class"""

from abc import ABC, abstractmethod
from typing import Dict

import numpy as np

from src.data.data_factory import DataFactory, Set


class EmotionClassifier(ABC):
    """
    This class is the base class for all emotion classifiers
    """

    def __init__(
        self,
        name: str = "base",
        data_type: str = None,
        parameters: Dict = None,
    ) -> None:
        """
        The initializer storing general initialization data

        :param name: The name of the classifier to distinguish
        :param data_type: The data type (text, image, audio, ...)
        :param parameters: Parameter dictionary containing all parameters
        """
        parameters = parameters or {}
        self.name = name
        self.data_type = data_type
        self.parameters = parameters
        self.data_reader = DataFactory.get_data_reader(data_type)
        self.emotions = parameters.get("emotions", "neutral_ekman")
        self.is_trained = False
        self.logger = None

    @abstractmethod
    def train(self, parameters: Dict, **kwargs) -> None:
        """
        The virtual training method for interfacing

        :param parameters: Parameter dictionary used for training
        :param kwargs: Additional kwargs parameters
        """
        raise NotImplementedError("Abstract class")  # pragma: no cover

    @abstractmethod
    def load(self, parameters: dict, **kwargs) -> None:
        """
        Loading method that loads a previously trained model from disk.

        :param parameters: Parameters required for loading the model
        :param kwargs: Additional kwargs parameters
        """
        raise NotImplementedError("Abstract class")  # pragma: no cover

    @abstractmethod
    def save(self, parameters: dict, **kwargs) -> None:
        """
        Saving method that saves a previously trained model on disk.

        :param parameters: Parameters required for storing the model
        :param kwargs: Additional kwargs parameters
        """
        raise NotImplementedError("Abstract class")  # pragma: no cover

    @abstractmethod
    def classify(self, parameters: Dict, **kwargs) -> np.array:
        """
        The virtual classification method for interfacing

        :param parameters: Parameter dictionary used for classification
        :param kwargs: Additional kwargs parameters
        :return: An array with predicted emotion indices
        """
        raise NotImplementedError("Abstract class")  # pragma: no cover

    def get_class_weights(
        self, which_set: Set, parameters: Dict = None
    ) -> Dict[int, int]:
        """
        Function that returns a class weights dictionary for a given dataset.
        The dictionary's keys are the labels and the values are the counts.

        :param which_set: Which set to use for calculating the class weights.
        :param parameters: Parameter dictionary
        :return: Dictionary with the class weights
        """
        labels = self.data_reader.get_labels(which_set, parameters)
        total = labels.shape[0]
        count_arr = np.bincount(labels.astype(int))
        return {
            index: (total / 7.0 / count)
            for index, count in enumerate(count_arr)
        }

    @staticmethod
    def init_parameters(parameters: Dict = None, **kwargs) -> Dict:
        """
        Function that merges the parameters and kwargs

        :param parameters: Parameter dictionary
        :param kwargs: Additional parameters in kwargs
        :return: Combined dictionary with parameters
        """
        parameters = parameters or {}
        parameters.update(kwargs)
        return parameters
