""" This file implements a Resnet classifier for the MFCC features derived
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
        l1 = parameters.get("l1", 1e-4)
        l2 = parameters.get("l2", 1e-3)
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
    }
    if (
        not os.path.exists("models/plant/plant_mfcc_resnet")
        or "train" in sys.argv
    ):
        accuracies = []
        per_class_accuracies = []
        for i in range(1, 5):
            this_acc = 0.0
            this_pc_acc = 0.0
            while this_acc < 0.3 or this_pc_acc < 0.3:
                cv_params = copy.deepcopy(parameters)
                classifier = PlantMFCCResnetClassifier()
                cv_params["cv_index"] = i
                classifier.train(cv_params)
                classifier.load({"save_path": "models/plant/checkpoint"})
                classifier.save(
                    {"save_path": f"models/plant/plant_mfcc_resnet_{i}"}
                )
                classifier.load({"save_path": "models/plant/checkpoint"})
                pred = classifier.classify(cv_params)
                labels = classifier.data_reader.get_labels(Set.TEST, cv_params)
                this_acc = accuracy(labels, pred)
                this_pc_acc = per_class_accuracy(labels, pred)
                classifier.data_reader.cleanup()
            accuracies.append(this_acc)
            per_class_accuracies.append(this_pc_acc)
        print(f"Training Acc: {np.mean(accuracies)} | {accuracies}")
        print(
            f"Training Class Acc: {np.mean(per_class_accuracies)} | "
            f"{per_class_accuracies}"
        )

    classifier.load(parameters)
    emotions = classifier.classify(parameters)
    print(np.unique(emotions, return_counts=True))
    labels = classifier.data_reader.get_labels(Set.TEST, parameters)
    print(f"Labels Shape: {labels.shape}")
    print(f"Emotions Shape: {emotions.shape}")
    print(f"Accuracy: {accuracy(labels, emotions)}")
    print(f"Per Class Accuracy: {per_class_accuracy(labels, emotions)}")
