""" This file implements a Fully Connected classifier for the plant data. """
import os
import sys
from typing import Dict

import tensorflow as tf

from src.classification.plant.nn_classifier import PlantNNBaseClassifier
from src.data.data_reader import Set
from src.utils.metrics import accuracy, per_class_accuracy


class PlantDenseClassifier(PlantNNBaseClassifier):
    def __init__(self, parameters: Dict = None):
        """
        Initialize the Plant-Dense emotion classifier

        :param name: The name for the classifier
        :param parameters: Some configuration parameters for the classifier
        """
        super().__init__("plant_dense", parameters)

    def initialize_model(self, parameters: Dict) -> None:
        """
        Initializes a new and pretrained version of the Plant-Dense model
        """
        dense_units = parameters.get("dense_units", 512)
        dropout = parameters.get("dropout", 0.2)
        input_size = self.data_reader.get_input_shape(parameters)[0]
        hidden_layers = parameters.get("hidden_layers", 2)
        input = tf.keras.layers.Input(
            shape=(input_size,), dtype=tf.float32, name="raw"
        )
        hidden = tf.keras.layers.Dense(dense_units)(input)
        hidden = tf.keras.layers.Dropout(dropout)(hidden)
        for layer_id in range(hidden_layers - 1):
            hidden = tf.keras.layers.Dense(dense_units)(hidden)
            hidden = tf.keras.layers.Dropout(dropout)(hidden)
        out = tf.keras.layers.Dense(7, activation="softmax")(hidden)
        self.model = tf.keras.Model(input, out)


if __name__ == "__main__":  # pragma: no cover
    classifier = PlantDenseClassifier()
    parameters = {
        "epochs": 10,
        "patience": 5,
        "learning_rate": 0.001,
        "dense_units": 1024,
        "hidden_layers": 2,
        "dropout": 0.2,
        "batch_size": 32,
    }
    if not os.path.exists("models/plant/plant_dense") or "train" in sys.argv:
        classifier.train(parameters)
        classifier.save()

    classifier.load(parameters)
    emotions = classifier.classify(parameters)
    labels = classifier.data_reader.get_labels(Set.TEST)
    print(f"Labels Shape: {labels.shape}")
    print(f"Emotions Shape: {emotions.shape}")
    print(f"Accuracy: {accuracy(labels, emotions)}")
    print(f"Per Class Accuracy: {per_class_accuracy(labels, emotions)}")
