""" This file implements an LSTM based classifier for the plant data. """

from typing import Dict

import tensorflow as tf

from src.classification.plant.nn_classifier import PlantNNBaseClassifier
from src.utils import cv_training_loop


class PlantLSTMClassifier(PlantNNBaseClassifier):
    """
    Classifier that uses LSTM layers and a Dense head for classification.
    """

    def __init__(self, parameters: Dict = None):
        """
        Initialize the Plant-LSTM emotion classifier

        :param parameters: Some configuration parameters for the classifier
        """
        super().__init__("plant_lstm", parameters)

    def initialize_model(self, parameters: Dict) -> None:
        """
        Initializes a new and pretrained version of the Plant-LSTM model

        :param parameters: Parameters for initializing the model.
        """
        lstm_units = parameters.get("lstm_units", 512)
        dropout = parameters.get("dropout", 0.2)
        l1 = parameters.get("l1", 1e-4)
        l2 = parameters.get("l2", 1e-3)
        regularizer = tf.keras.regularizers.L1L2(l1=l1, l2=l2)
        input_size = self.data_reader.get_input_shape(parameters)[0]
        lstm_layers = parameters.get("lstm_layers", 1)
        input_tensor = tf.keras.layers.Input(
            shape=(input_size, 1), dtype=tf.float32, name="raw"
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
        out = tf.keras.layers.Dense(
            1024, activation="relu", kernel_regularizer=regularizer
        )(out)
        out = tf.keras.layers.Dropout(dropout)(out)
        out = tf.keras.layers.Dense(
            512, activation="relu", kernel_regularizer=regularizer
        )(out)
        out = tf.keras.layers.Dropout(dropout)(out)
        out = tf.keras.layers.Dense(7, activation="softmax")(out)
        self.model = tf.keras.Model(input_tensor, out)


def _main():  # pragma: no cover
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
    save_path = "models/plant/plant_lstm"
    cv_training_loop(classifier, parameters, save_path)


if __name__ == "__main__":  # pragma: no cover
    _main()
