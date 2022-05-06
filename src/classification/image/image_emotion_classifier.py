""" Base class for all image emotion classifiers """

from abc import abstractmethod
from typing import Dict

import numpy as np
import tensorflow as tf

from src.classification import EmotionClassifier
from src.data.data_reader import Set


class ImageEmotionClassifier(EmotionClassifier):
    """
    Base class for all image emotion classifiers. Contains common functionality
    that concerns all image classifiers.
    """

    def __init__(self, name: str = "image", parameters: Dict = None):
        """
        Initialize the Image emotion classifier

        :param name: The name for the classifier
        :param parameters: Some configuration parameters for the classifier
        """
        super().__init__(name, "image", parameters)
        self.callback = None
        self.optimizer = None
        self.loss = None
        self.metrics = None
        self.train_data = None
        self.val_data = None
        self.class_weights = None

    @abstractmethod
    def train(self, parameters: Dict = None, **kwargs) -> None:
        """
        Virtual training method for interfacing

        :param parameters: Parameter dictionary used for training
        :param kwargs: Additional kwargs parameters
        """
        raise NotImplementedError("Abstract class")  # pragma: no cover

    @abstractmethod
    def load(self, parameters: Dict = None, **kwargs) -> None:
        """
        Loading method that loads a previously trained model from disk.

        :param parameters: Parameters required for loading the model
        :param kwargs: Additional kwargs parameters
        """
        raise NotImplementedError("Abstract class")  # pragma: no cover

    @abstractmethod
    def save(self, parameters: Dict = None, **kwargs) -> None:
        """
        Saving method that saves a previously trained model on disk.

        :param parameters: Parameters required for storing the model
        :param kwargs: Additional kwargs parameters
        """
        raise NotImplementedError("Abstract class")  # pragma: no cover

    @abstractmethod
    def classify(self, parameters: Dict = None, **kwargs) -> np.array:
        """
        The virtual classification method for interfacing

        :param parameters: Parameter dictionary used for classification
        :param kwargs: Additional kwargs parameters
        :return: An array with predicted emotion indices
        """
        raise NotImplementedError("Abstract class")  # pragma: no cover

    def prepare_training(self, parameters: Dict) -> None:
        """
        Function that prepares the training by initializing optimizer,
        loss, metrics and callbacks for training.

        :param parameters: Training parameters
        """
        learning_rate = parameters.get("learning_rate", 0.001)
        patience = parameters.get("patience", 5)

        self.callback = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=patience, restore_best_weights=True
        )
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.metrics = [tf.metrics.CategoricalAccuracy()]
        self.loss = tf.keras.losses.CategoricalCrossentropy()

    def prepare_data(self, parameters: Dict) -> None:
        """
        Function that prepares image datasets for training and stores them
        inside the class.

        :param parameters: Parameter dictionary that contains important params.
            including: which_set, batch_size, weighted
        """
        which_set = parameters.get("which_set", Set.TRAIN)
        batch_size = parameters.get("batch_size", 64)
        weighted = parameters.get("weighted", True)

        self.train_data = self.data_reader.get_emotion_data(
            self.emotions, which_set, batch_size, parameters
        ).map(lambda x, y: (tf.image.grayscale_to_rgb(x), y))
        self.val_data = self.data_reader.get_emotion_data(
            self.emotions, Set.VAL, batch_size
        ).map(lambda x, y: (tf.image.grayscale_to_rgb(x), y))
        if weighted:
            self.class_weights = self.get_class_weights(which_set)
        else:
            self.class_weights = None
