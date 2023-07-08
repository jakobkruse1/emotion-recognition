""" This file implements a Resnet classifier for the MFCC features derived
from the plant data. """

import os
from typing import Dict

import tensorflow as tf

from src.classification.plant.nn_classifier import PlantNNBaseClassifier
from src.utils import cv_training_loop


class PlantMFCCResnetClassifier(PlantNNBaseClassifier):
    """
    Model that uses MFCC features and a resnet50 classifier.
    """

    def __init__(self, parameters: Dict = None):
        """
        Initialize the Plant-MFCC-Resnet emotion classifier

        :param parameters: Some configuration parameters for the classifier
        """
        super().__init__("plant_mfcc_resnet", parameters)

    def initialize_model(self, parameters: Dict) -> None:
        """
        Initializes a new and pretrained version of the Plant-MFCC-Resnet model

        :param parameters: Parameters for initializing the model
        """
        dropout = parameters.get("dropout", 0.2)
        num_mfcc = parameters.get("num_mfcc", 40)
        pretrained = parameters.get("pretrained", True)
        l1 = parameters.get("l1", 1e-4)
        l2 = parameters.get("l2", 1e-3)
        input_size = self.data_reader.get_input_shape(parameters)[0]
        input_tensor = tf.keras.layers.Input(
            shape=(input_size,), dtype=tf.float32, name="raw"
        )
        mfcc = self.compute_mfccs(input_tensor, {"num_mfcc": num_mfcc})
        mfcc = tf.expand_dims(mfcc, 3)
        mfcc = tf.stack([mfcc, mfcc, mfcc], axis=3)
        resnet = tf.keras.applications.resnet.ResNet50(
            include_top=False,
            weights="imagenet" if pretrained else None,
            pooling=None,
        )
        regularizer = tf.keras.regularizers.L1L2(l1=l1, l2=l2)
        for layer in resnet.layers:
            for attr in ["kernel_regularizer", "bias_regularizer"]:
                if hasattr(layer, attr):
                    setattr(layer, attr, regularizer)
        resnet_out = resnet(mfcc)

        hidden = tf.keras.layers.Flatten()(resnet_out)
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
    classifier = PlantMFCCResnetClassifier()
    parameters = {
        "epochs": 1000,
        "patience": 100,
        "batch_size": 64,
        "preprocess": False,
        "learning_rate": 0.001,
        "dropout": 0.2,
        "label_mode": "both",
        "pretrained": False,
        "num_mfcc": 60,
        "window": 20,
        "hop": 10,
        "balanced": True,
        "checkpoint": True,
    }
    save_path = os.path.join("models", "plant", "plant_mfcc_resnet")
    cv_training_loop(classifier, parameters, save_path)


if __name__ == "__main__":  # pragma: no cover
    _main()
