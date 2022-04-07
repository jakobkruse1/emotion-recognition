"""Implement an emotion classifier base class"""

from abc import ABC, abstractmethod
from typing import Dict

import numpy as np

from src.data.data_factory import DataFactory


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