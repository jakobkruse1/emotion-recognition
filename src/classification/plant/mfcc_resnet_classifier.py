""" This file implements a Resnet classifier for the MFCC features derived
from the plant data. """
import os
import sys
from typing import Dict

import numpy as np
import tensorflow as tf

from src.classification.plant.nn_classifier import PlantNNBaseClassifier
from src.data.data_reader import Set
from src.utils.metrics import accuracy, per_class_accuracy


class PlantMFCCResnetClassifier(PlantNNBaseClassifier):
    def __init__(self, parameters: Dict = None):
        """
        Initialize the Plant-MFCC-Resnet emotion classifier

        :param name: The name for the classifier
        :param parameters: Some configuration parameters for the classifier
        """
        super().__init__("plant_mfcc_resnet", parameters)

    def initialize_model(self, parameters: Dict) -> None:
        """
        Initializes a new and pretrained version of the Plant-MFCC-Resnet model
        """
        dropout = parameters.get("dropout", 0.2)
        num_mfcc = parameters.get("num_mfcc", 40)
        pretrained = parameters.get("pretrained", True)
        input_size = self.data_reader.get_input_shape(parameters)[0]
        input = tf.keras.layers.Input(
            shape=(input_size,), dtype=tf.float32, name="raw"
        )
        mfcc = self.compute_mfccs(input, {"num_mfcc": num_mfcc})
        mfcc = tf.expand_dims(mfcc, 3)
        mfcc = tf.stack([mfcc, mfcc, mfcc], axis=3)
        resnet = tf.keras.applications.resnet.ResNet50(
            include_top=False,
            weights="imagenet" if pretrained else None,
            pooling=None,
        )(mfcc)

        hidden = tf.keras.layers.Flatten()(resnet)
        hidden = tf.keras.layers.Dense(1024)(hidden)
        hidden = tf.keras.layers.Dropout(dropout)(hidden)
        out = tf.keras.layers.Dense(7, activation="softmax")(hidden)
        self.model = tf.keras.Model(input, out)

    @staticmethod
    def init_parameters(parameters: Dict = None, **kwargs) -> Dict:
        """
        Function that merges the parameters and kwargs

        :param parameters: Parameter dictionary
        :param kwargs: Additional parameters in kwargs
        :return: Combined dictionary with parameters
        """
        parameters = parameters or {}
        parameters.update(kwargs)
        parameters["preprocess"] = False  # Crucial step otherwise MFCC broken
        return parameters


if __name__ == "__main__":  # pragma: no cover
    classifier = PlantMFCCResnetClassifier()
    parameters = {
        "epochs": 50,
        "patience": 10,
        "batch_size": 32,
        "window": 20,
        "hop": 20,
        "label_mode": "both",
        "balanced": True,
    }
    if (
        not os.path.exists("models/plant/plant_mfcc_resnet")
        or "train" in sys.argv
    ):
        classifier.train(parameters)
        classifier.save()

    classifier.load(parameters)
    emotions = classifier.classify(parameters)
    print(np.unique(emotions, return_counts=True))
    labels = classifier.data_reader.get_labels(Set.TEST, parameters)
    print(f"Labels Shape: {labels.shape}")
    print(f"Emotions Shape: {emotions.shape}")
    print(f"Accuracy: {accuracy(labels, emotions)}")
    print(f"Per Class Accuracy: {per_class_accuracy(labels, emotions)}")
