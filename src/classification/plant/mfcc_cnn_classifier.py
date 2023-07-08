""" This file implements a CNN classifier for the MFCC features derived
from the plant data. """

import os
from typing import Dict

import tensorflow as tf

from src.classification.plant.nn_classifier import PlantNNBaseClassifier
from src.utils import cv_training_loop


class PlantMFCCCNNClassifier(PlantNNBaseClassifier):
    """
    Model that uses MFCC features and Conv layers for classification.
    """

    def __init__(self, parameters: Dict = None):
        """
        Initialize the Plant-MFCC-CNN emotion classifier

        :param parameters: Some configuration parameters for the classifier
        """
        super().__init__("plant_mfcc_cnn", parameters)

    def initialize_model(self, parameters: Dict) -> None:
        """
        Initializes a new and pretrained version of the Plant-MFCC-CNN model

        :param parameters: Parameters for initializing the model
        """
        dropout = parameters.get("dropout", 0.2)
        conv_layers = parameters.get("conv_layers", 3)
        conv_filters = parameters.get("conv_filters", 64)
        conv_kernel_size = parameters.get("conv_kernel_size", 7)
        l1 = parameters.get("l1", 1e-4)
        l2 = parameters.get("l2", 1e-3)
        input_size = self.data_reader.get_input_shape(parameters)[0]
        input_tensor = tf.keras.layers.Input(
            shape=(input_size,), dtype=tf.float32, name="raw"
        )
        regularizer = tf.keras.regularizers.L1L2(l1=l1, l2=l2)
        mfcc = self.compute_mfccs(input_tensor)
        hidden = tf.expand_dims(mfcc, 3)
        for i in range(conv_layers):
            hidden = tf.keras.layers.Conv2D(
                conv_filters,
                kernel_size=conv_kernel_size,
                padding="same",
                kernel_regularizer=regularizer,
            )(hidden)
            hidden = tf.keras.layers.MaxPooling2D()(hidden)

        hidden = tf.keras.layers.Flatten()(hidden)
        hidden = tf.keras.layers.Dense(1024, kernel_regularizer=regularizer)(
            hidden
        )
        hidden = tf.keras.layers.Dropout(dropout)(hidden)
        hidden = tf.keras.layers.Dense(1024, kernel_regularizer=regularizer)(
            hidden
        )
        hidden = tf.keras.layers.Dropout(dropout)(hidden)
        out = tf.keras.layers.Dense(7, activation="softmax")(hidden)
        self.model = tf.keras.Model(input_tensor, out)

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


def _main():  # pragma: no cover
    classifier = PlantMFCCCNNClassifier()
    parameters = {
        "epochs": 1000,
        "patience": 100,
        "batch_size": 64,
        "preprocess": False,
        "learning_rate": 0.0003,
        "conv_filters": 96,
        "conv_layers": 2,
        "conv_kernel_size": 7,
        "dropout": 0.2,
        "label_mode": "both",
        "window": 20,
        "hop": 10,
        "balanced": True,
        "checkpoint": True,
    }
    save_path = os.path.join("models", "plant", "plant_mfcc_cnn")
    cv_training_loop(classifier, parameters, save_path)


if __name__ == "__main__":  # pragma: no cover
    _main()
