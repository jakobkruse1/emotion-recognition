""" This file implements a Fully Connected classifier for the plant data. """

from typing import Dict

import tensorflow as tf

from src.classification.plant.nn_classifier import PlantNNBaseClassifier
from src.utils import cv_training_loop


class PlantDenseClassifier(PlantNNBaseClassifier):
    """
    Classifier consisting of Dense layers only.
    """

    def __init__(self, parameters: Dict = None):
        """
        Initialize the Plant-Dense emotion classifier

        :param parameters: Some configuration parameters for the classifier
        """
        super().__init__("plant_dense", parameters)

    def initialize_model(self, parameters: Dict) -> None:
        """
        Initializes a new and pretrained version of the Plant-Dense model

        :param parameters: Parameters for initializing the model.
        """
        dense_units = parameters.get("dense_units", 512)
        dropout = parameters.get("dropout", 0.2)
        l1 = parameters.get("l1", 1e-4)
        l2 = parameters.get("l2", 1e-3)
        regularizer = tf.keras.regularizers.L1L2(l1=l1, l2=l2)
        input_size = self.data_reader.get_input_shape(parameters)[0]
        hidden_layers = parameters.get("hidden_layers", 2)
        input_tensor = tf.keras.layers.Input(
            shape=(input_size,), dtype=tf.float32, name="raw"
        )
        hidden = tf.keras.layers.Dense(
            dense_units, kernel_regularizer=regularizer, activation="relu"
        )(input_tensor)
        hidden = tf.keras.layers.Dropout(dropout)(hidden)
        for layer_id in range(hidden_layers - 1):
            hidden = tf.keras.layers.Dense(
                dense_units, kernel_regularizer=regularizer, activation="relu"
            )(hidden)
            hidden = tf.keras.layers.Dropout(dropout)(hidden)
        out = tf.keras.layers.Dense(7, activation="softmax")(hidden)
        self.model = tf.keras.Model(input_tensor, out)


def _main():  # pragma: no cover
    classifier = PlantDenseClassifier()
    parameters = {
        "epochs": 50,
        "patience": 10,
        "batch_size": 64,
        "learning_rate": 0.001,
        "dense_units": 4096,
        "downsampling_factor": 500,
        "dense_layers": 2,
        "dropout": 0.2,
        "label_mode": "both",
        "window": 20,
        "hop": 10,
        "balanced": True,
    }
    save_path = "models/plant/plant_dense"
    cv_training_loop(classifier, parameters, save_path)


if __name__ == "__main__":  # pragma: no cover
    _main()
