""" This file implements an LSTM based classifier for the plant data. """

from abc import abstractmethod
from typing import Dict

import numpy as np
import tensorflow as tf

from src.classification.plant.plant_emotion_classifier import (
    PlantEmotionClassifier,
)
from src.data.data_reader import Set


class PlantNNBaseClassifier(PlantEmotionClassifier):
    def __init__(self, name: str, parameters: Dict = None):
        """
        Initialize the Plant-LSTM emotion classifier

        :param name: The name for the classifier
        :param parameters: Some configuration parameters for the classifier
        """
        super().__init__(name, parameters)
        tf.get_logger().setLevel("ERROR")
        self.model = None

    @abstractmethod
    def initialize_model(self, parameters: Dict) -> None:
        """
        Base class that creates self.model, a tf Model instance
        """
        raise NotImplementedError()

    def train(self, parameters: Dict = None, **kwargs) -> None:
        """
        Training method for Plant-LSTM model

        :param parameters: Parameter dictionary used for training
        :param kwargs: Additional kwargs parameters
        """
        parameters = self.init_parameters(parameters, **kwargs)
        epochs = parameters.get("epochs", 20)

        if not self.model:
            self.initialize_model(parameters)
        self.prepare_training(parameters)
        self.model.compile(
            optimizer=self.optimizer, loss=self.loss, metrics=self.metrics
        )
        self.prepare_data(parameters)

        _ = self.model.fit(
            x=self.train_data,
            validation_data=self.val_data,
            epochs=epochs,
            callbacks=[self.callback],
            class_weight=self.class_weights,
        )
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

    def classify(self, parameters: Dict = None, **kwargs) -> np.array:
        """
        Classification method used to classify emotions from speech

        :param parameters: Parameter dictionary used for classification
        :param kwargs: Additional kwargs parameters
        :return: An array with predicted emotion indices
        """
        parameters = self.init_parameters(parameters, **kwargs)
        which_set = parameters.get("which_set", Set.TEST)
        batch_size = parameters.get("batch_size", 64)
        eval_parameters = parameters.copy()
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