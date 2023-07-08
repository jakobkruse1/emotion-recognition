""" This file implements an LSTM based classifier for the watch data. """

import os
from typing import Dict

import tensorflow as tf

from src.classification.watch.nn_classifier import WatchNNBaseClassifier
from src.utils import cv_training_loop


class WatchLSTMClassifier(WatchNNBaseClassifier):
    """
    Classifier that uses LSTM layers and a Dense head for classification.
    """

    def __init__(self, parameters: Dict = None):
        """
        Initialize the Watch-LSTM emotion classifier

        :param parameters: Some configuration parameters for the classifier
        """
        super().__init__("watch_lstm", parameters)

    def initialize_model(self, parameters: Dict) -> None:
        """
        Initializes a new and pretrained version of the Watch-LSTM model

        :param parameters: Parameters for initializing the model.
        """
        lstm_units = parameters.get("lstm_units", 512)
        dropout = parameters.get("dropout", 0.2)
        input_size = self.data_reader.get_input_shape(parameters)
        lstm_layers = parameters.get("lstm_layers", 1)
        input_tensor = tf.keras.layers.Input(
            shape=(*input_size,), dtype=tf.float32, name="raw"
        )
        if lstm_layers >= 2:
            out = tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(lstm_units, return_sequences=True)
            )(input_tensor)
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
            )(input_tensor)
        out = tf.keras.layers.Dropout(dropout)(out)
        out = tf.keras.layers.Dense(1024, activation="relu")(out)
        out = tf.keras.layers.Dropout(dropout)(out)
        out = tf.keras.layers.Dense(512, activation="relu")(out)
        out = tf.keras.layers.Dropout(dropout)(out)
        out = tf.keras.layers.Dense(7, activation="softmax")(out)
        self.model = tf.keras.Model(input_tensor, out)


def _main():  # pragma: no cover
    classifier = WatchLSTMClassifier()
    parameters = {
        "epochs": 1000,
        "patience": 100,
        "batch_size": 64,
        "window": 20,
        "hop": 2,
        "balanced": True,
        "learning_rate": 0.001,
        "lstm_units": 4096,
        "dropout": 0.2,
        "lstm_layers": 1,
    }
    save_path = os.path.join("models", "watch", "watch_lstm")
    cv_training_loop(classifier, parameters, save_path)


if __name__ == "__main__":  # pragma: no cover
    _main()
