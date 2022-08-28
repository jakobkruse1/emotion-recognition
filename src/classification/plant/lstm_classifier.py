""" This file implements an LSTM based classifier for the plant data. """
import os
import sys
from typing import Dict

import tensorflow as tf

from src.classification.plant.nn_classifier import PlantNNBaseClassifier
from src.data.data_reader import Set
from src.utils.metrics import accuracy, per_class_accuracy


class PlantLSTMClassifier(PlantNNBaseClassifier):
    def __init__(self, parameters: Dict = None):
        """
        Initialize the Plant-LSTM emotion classifier

        :param name: The name for the classifier
        :param parameters: Some configuration parameters for the classifier
        """
        super().__init__("plant_lstm", parameters)

    def initialize_model(self, parameters: Dict) -> None:
        """
        Initializes a new and pretrained version of the Plant-LSTM model
        """
        lstm_units = parameters.get("lstm_units", 512)
        dropout = parameters.get("dropout", 0.2)
        input_size = self.data_reader.get_input_shape(parameters)[0]
        lstm_layers = parameters.get("lstm_layers", 1)
        input = tf.keras.layers.Input(
            shape=(input_size, 1), dtype=tf.float32, name="raw"
        )
        if lstm_layers >= 2:
            out = tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(lstm_units, return_sequences=True)
            )(input)
            for i in range(lstm_layers - 2):
                out = tf.keras.layers.Bidirectional(
                    tf.keras.layers.LSTM(lstm_units, return_sequences=True)
                )(out)
            out = tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(lstm_units)
            )(out)
        else:
            out = tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(lstm_units)
            )(input)
        out = tf.keras.layers.Dropout(dropout)(out)
        out = tf.keras.layers.Dense(1024, activation="relu")(out)
        out = tf.keras.layers.Dropout(dropout)(out)
        out = tf.keras.layers.Dense(512, activation="relu")(out)
        out = tf.keras.layers.Dropout(dropout)(out)
        out = tf.keras.layers.Dense(7, activation="softmax")(out)
        self.model = tf.keras.Model(input, out)


if __name__ == "__main__":  # pragma: no cover
    classifier = PlantLSTMClassifier()
    parameters = {
        "epochs": 50,
        "patience": 10,
        "batch_size": 64,
        "learning_rate": 0.0003,
        "lstm_units": 1024,
        "lstm_layers": 2,
        "dropout": 0,
        "label_mode": "faceapi",
        "window": 30,
        "hop": 30,
        "balanced": True,
    }
    if not os.path.exists("models/plant/plant_lstm") or "train" in sys.argv:
        classifier.train(parameters)
        classifier.save()

    classifier.load(parameters)
    emotions = classifier.classify(parameters)
    labels = classifier.data_reader.get_labels(Set.TEST)
    print(f"Labels Shape: {labels.shape}")
    print(f"Emotions Shape: {emotions.shape}")
    print(f"Accuracy: {accuracy(labels, emotions)}")
    print(f"Per Class Accuracy: {per_class_accuracy(labels, emotions)}")
