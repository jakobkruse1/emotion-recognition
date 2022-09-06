""" This file implements a CNN classifier for the MFCC features derived
from the plant data. """
import copy
import os
import sys
from typing import Dict

import numpy as np
import tensorflow as tf

from src.classification.plant.nn_classifier import PlantNNBaseClassifier
from src.data.data_reader import Set
from src.utils.metrics import accuracy, per_class_accuracy


class PlantMFCCCNNClassifier(PlantNNBaseClassifier):
    def __init__(self, parameters: Dict = None):
        """
        Initialize the Plant-MFCC-CNN emotion classifier

        :param name: The name for the classifier
        :param parameters: Some configuration parameters for the classifier
        """
        super().__init__("plant_mfcc_cnn", parameters)

    def initialize_model(self, parameters: Dict) -> None:
        """
        Initializes a new and pretrained version of the Plant-MFCC-CNN model
        """
        dropout = parameters.get("dropout", 0.2)
        conv_layers = parameters.get("conv_layers", 3)
        conv_filters = parameters.get("conv_filters", 64)
        conv_kernel_size = parameters.get("conv_kernel_size", 7)
        l1 = parameters.get("l1", 1e-4)
        l2 = parameters.get("l2", 1e-3)
        input_size = self.data_reader.get_input_shape(parameters)[0]
        input = tf.keras.layers.Input(
            shape=(input_size,), dtype=tf.float32, name="raw"
        )
        regularizer = tf.keras.regularizers.L1L2(l1=l1, l2=l2)
        mfcc = self.compute_mfccs(input)
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
    classifier = PlantMFCCCNNClassifier()
    parameters = {
        "epochs": 50,
        "patience": 10,
        "batch_size": 64,
        "preprocess": False,
        "learning_rate": 0.0003,
        "conv_filters": 96,
        "conv_layers": 2,
        "conv_kernel_size": 7,
        "dropout": 0.2,
        "label_mode": "both",
        "window": 10,
        "hop": 10,
        "weighted": True,
    }
    if (
        not os.path.exists("models/plant/plant_mfcc_cnn")
        or "train" in sys.argv
    ):
        accuracies = []
        per_class_accuracies = []
        for i in range(5):
            cv_params = copy.deepcopy(parameters)
            cv_params["cv_index"] = i
            classifier.train(cv_params)
            if i == 0:
                classifier.save()
            classifier.load({"save_path": "models/plant/checkpoint"})
            pred = classifier.classify(cv_params)
            labels = classifier.data_reader.get_labels(Set.TEST, cv_params)
            accuracies.append(accuracy(labels, pred))
            per_class_accuracies.append(per_class_accuracy(labels, pred))
        print(f"Training Acc: {np.mean(accuracies)} | {accuracies}")
        print(
            f"Training Class Acc: {np.mean(per_class_accuracies)} | "
            f"{per_class_accuracies}"
        )

    classifier.load(parameters)
    emotions = classifier.classify(parameters)
    labels = classifier.data_reader.get_labels(Set.TEST, parameters)
    print(f"Labels Shape: {labels.shape}")
    print(f"Emotions Shape: {emotions.shape}")
    print(f"Accuracy: {accuracy(labels, emotions)}")
    print(f"Per Class Accuracy: {per_class_accuracy(labels, emotions)}")
