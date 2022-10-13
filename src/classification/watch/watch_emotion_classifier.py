""" Base class for all watch emotion classifiers """

from abc import abstractmethod
from typing import Dict

import numpy as np
import tensorflow as tf

from src.classification import EmotionClassifier
from src.data.data_factory import DataFactory
from src.data.data_reader import Set


class WatchEmotionClassifier(EmotionClassifier):
    """
    Base class for all watch emotion classifiers. Contains common functions
    that concerns all smartwatch classifiers.
    """

    def __init__(self, name: str = "watch", parameters: Dict = None):
        """
        Initialize the plant emotion classifier

        :param name: The name for the classifier
        :param parameters: Some configuration parameters for the classifier
        """
        super().__init__(name, "watch", parameters)
        self.callbacks = None
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

        self.callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=patience,
                restore_best_weights=True,
            )
        ]
        if parameters.get("checkpoint", False):
            self.callbacks.append(
                tf.keras.callbacks.ModelCheckpoint(
                    f"models/watch/checkpoint_{parameters.get('cv_index', '')}",
                    save_best_only=True,
                    monitor="val_categorical_accuracy",
                    mode="max",
                )
            )
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.metrics = [tf.metrics.CategoricalAccuracy()]
        self.loss = tf.keras.losses.CategoricalCrossentropy()

    def prepare_data(self, parameters: Dict) -> None:
        """
        Function that prepares speech datasets for training and stores them
        inside the class.

        :param parameters: Parameter dictionary that contains important params.
            including: which_set, batch_size, weighted
        """
        which_set = parameters.get("which_set", Set.TRAIN)
        batch_size = parameters.get("batch_size", 64)
        weighted = parameters.get("weighted", False)

        balanced = parameters.get("balanced", False)
        if balanced:
            self.data_reader = DataFactory.get_data_reader(
                "balanced_watch", self.data_reader.folder
            )

        self.train_data = self.data_reader.get_emotion_data(
            self.emotions, which_set, batch_size, parameters
        )
        self.val_data = self.data_reader.get_emotion_data(
            self.emotions, Set.VAL, batch_size, parameters
        )
        if weighted:
            self.class_weights = self.get_class_weights(which_set, parameters)
        else:
            self.class_weights = None


def _main():  # pragma: no cover
    from src.data.watch_exp_reader import WatchExperimentDataReader

    dr = WatchExperimentDataReader()

    for watch, labels in dr.get_seven_emotion_data(
        Set.TEST, parameters={}
    ).take(1):
        print(watch.shape)
        print(labels.shape)


if __name__ == "__main__":  # pragma: no cover
    _main()
