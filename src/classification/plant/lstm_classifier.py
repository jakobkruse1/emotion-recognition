""" This file implements an LSTM based classifier for the plant data. """
import os
import sys
from typing import Dict

import numpy as np
import tensorflow as tf

from src.classification.plant.plant_emotion_classifier import (
    PlantEmotionClassifier,
)
from src.data.data_reader import Set


class PlantLSTMClassifier(PlantEmotionClassifier):
    def __init__(self, parameters: Dict = None):
        """
        Initialize the Plant-LSTM emotion classifier

        :param name: The name for the classifier
        :param parameters: Some configuration parameters for the classifier
        """
        super().__init__("plant_lstm", parameters)
        tf.get_logger().setLevel("ERROR")
        self.model = None

    def initialize_model(self, parameters: Dict) -> None:
        """
        Initializes a new and pretrained version of the Plant-LSTM model
        """
        lstm_units = parameters.get("lstm_units", 512)
        dropout = parameters.get("dropout", 0.2)
        window = parameters.get("window", 10)
        input = tf.keras.layers.Input(
            shape=(window * 10000, 1), dtype=tf.float32, name="raw"
        )
        out = tf.keras.layers.LSTM(lstm_units)(input)
        out = tf.keras.layers.Dropout(dropout)(out)
        out = tf.keras.layers.Dense(1024, activation="relu")(out)
        out = tf.keras.layers.Dense(512, activation="relu")(out)
        out = tf.keras.layers.Dense(7, activation="softmax")(out)
        self.model = tf.keras.Model(input, out)
        print(self.model.summary())

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
        save_path = parameters.get("save_path", "models/plant/plant_lstm")
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
        save_path = parameters.get("save_path", "models/plant/plant_lstm")
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
        dataset = self.data_reader.get_emotion_data(
            self.emotions, which_set, batch_size, parameters
        )

        if not self.model:
            raise RuntimeError(
                "Please load or train the model before inference!"
            )
        results = self.model.predict(dataset)
        return np.argmax(results, axis=1)


if __name__ == "__main__":  # pragma: no cover
    classifier = PlantLSTMClassifier()
    parameters = {
        "epochs": 10,
        "patience": 5,
        "learning_rate": 0.001,
        "lstm_units": 128,
        "dropout": 0.2,
        "batch_size": 32,
    }
    if not os.path.exists("models/plant/plant_lstm") or "train" in sys.argv:
        classifier.train(parameters)
        classifier.save()

    classifier.load(parameters)
    emotions = classifier.classify(parameters)
    labels = classifier.data_reader.get_labels(Set.TEST)
    print(f"Labels Shape: {labels.shape}")
    print(f"Emotions Shape: {emotions.shape}")
    print(f"Accuracy: {np.sum(emotions == labels) / labels.shape[0]}")
