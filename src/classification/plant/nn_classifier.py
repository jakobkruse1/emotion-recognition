""" This file defines an interface for classifiers for the plant data. """

import copy
from abc import abstractmethod
from typing import Dict

import numpy as np
import tensorflow as tf

from src.classification.plant.plant_emotion_classifier import (
    PlantEmotionClassifier,
)
from src.data.data_reader import Set
from src.utils import logging


class PlantNNBaseClassifier(PlantEmotionClassifier):
    """
    Base class for all NN classifiers in tensorflow for plant data.
    """

    def __init__(self, name: str, parameters: Dict = None):
        """
        Initialize the Plant classifier. All tensorflow plant classifiers
        require the same training, classification, save and load functions.
        This class implements these methods for all classifiers at once.

        :param name: The name for the classifier
        :param parameters: Some configuration parameters for the classifier
        """
        super().__init__(name, parameters)
        tf.get_logger().setLevel("ERROR")
        self.model = None
        self.logger = logging.KerasLogger()
        self.logger.log_start({"init_parameters": parameters})

    @abstractmethod
    def initialize_model(self, parameters: Dict) -> None:  # pragma: no cover
        """
        Abstract method that creates self.model, a tf Model instance

        :param parameters: Parameters for initializing the model
        """
        raise NotImplementedError()

    def train(self, parameters: Dict = None, **kwargs) -> None:
        """
        Training method for plant models

        :param parameters: Parameter dictionary used for training
        :param kwargs: Additional kwargs parameters
        """
        parameters = self.init_parameters(parameters, **kwargs)
        self.logger.log_start({"train_parameters": parameters})
        epochs = parameters.get("epochs", 20)

        if not self.model:
            self.initialize_model(parameters)
        self.prepare_training(parameters)
        self.model.compile(
            optimizer=self.optimizer, loss=self.loss, metrics=self.metrics
        )
        self.prepare_data(parameters)

        history = self.model.fit(
            x=self.train_data,
            validation_data=self.val_data,
            epochs=epochs,
            callbacks=self.callbacks,
            class_weight=self.class_weights,
        )
        self.logger.log_end({"history": history})
        self.is_trained = True

    def load(self, parameters: Dict = None, **kwargs) -> None:
        """
        Loading method that loads a previously trained model from disk.

        :param parameters: Parameters required for loading the model
        :param kwargs: Additional kwargs parameters
        """
        parameters = self.init_parameters(parameters, **kwargs)
        save_path = parameters.get("save_path", f"models/plant/{self.name}")
        self.model = tf.keras.models.load_model(save_path)

    def save(self, parameters: Dict = None, **kwargs) -> None:
        """
        Saving method that saves a previously trained model on disk.

        :param parameters: Parameters required for storing the model
        :param kwargs: Additional kwargs parameters
        """
        if not self.is_trained:
            raise RuntimeError(
                "Model needs to be trained in order to save it!"
            )
        parameters = self.init_parameters(parameters, **kwargs)
        save_path = parameters.get("save_path", f"models/plant/{self.name}")
        self.model.save(save_path, include_optimizer=False)
        self.logger.save_logs(save_path)

    def classify(self, parameters: Dict = None, **kwargs) -> np.array:
        """
        Classification method used to classify emotions from plant data

        :param parameters: Parameter dictionary used for classification
        :param kwargs: Additional kwargs parameters
        :return: An array with predicted emotion indices
        """
        parameters = self.init_parameters(parameters, **kwargs)
        which_set = parameters.get("which_set", Set.TEST)
        batch_size = parameters.get("batch_size", 64)
        eval_parameters = copy.deepcopy(parameters)
        eval_parameters["balanced"] = False
        dataset = self.data_reader.get_emotion_data(
            self.emotions, which_set, batch_size, eval_parameters
        )

        if not self.model:
            raise RuntimeError(
                "Please load or train the model before inference!"
            )
        results = self.model.predict(dataset)
        return np.argmax(results, axis=1)


if __name__ == "__main__":  # pragma: no cover
    classifier = PlantNNBaseClassifier("base")  # Does not work.
