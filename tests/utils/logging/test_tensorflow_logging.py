""" Test the Keras and tensorflow logger. """

import json
import os.path
import shutil
from typing import Any, Dict

import numpy as np
import tensorflow as tf

from src.classification.emotion_classifier import EmotionClassifier
from src.utils import logging


class NoopClassifier(EmotionClassifier):
    def __init__(self, parameters: Dict[str, Any]):
        super().__init__("noop", "text", parameters)
        self.logger = logging.KerasLogger()
        self.logger.log_start({"init_parameters": parameters})

    def train(self, parameters: Dict, **kwargs) -> None:
        self.logger.log_start({"train_parameters": parameters})
        for epoch in range(10):
            self.logger.log_epoch({"epoch_param": 1})
        model = tf.keras.models.Sequential(
            [tf.keras.layers.Dense(5, activation="softmax")]
        )
        model.compile(
            tf.keras.optimizers.SGD(),
            loss="categorical_crossentropy",
            metrics=[tf.metrics.CategoricalAccuracy()],
        )
        history = model.fit(
            np.arange(100).reshape(5, 20),
            np.eye(5, 5),
            validation_data=[np.arange(100).reshape(5, 20), np.eye(5, 5)],
            epochs=10,
            verbose=1,
        )
        self.logger.log_end({"history": history})

    def load(self, parameters: dict, **kwargs) -> None:
        pass

    def save(self, parameters: dict, **kwargs) -> None:
        folder = "models/temp/test_logging"
        self.logger.save_logs(folder)

    def classify(self, parameters: Dict, **kwargs) -> np.array:
        pass


def test_logging():
    shutil.rmtree("models/temp/test_logging", ignore_errors=True)

    classifier = NoopClassifier({"init1": 1, "init2": "testing"})
    classifier.train({"train1": 2.5, "train2": "train"})
    classifier.save({})

    assert os.path.exists("models/temp/test_logging/statistics.json")
    with open("models/temp/test_logging/statistics.json", "r") as json_file:
        logs = json.load(json_file)

    for key in [
        "train_parameters",
        "init_parameters",
        "train_loss",
        "val_loss",
        "train_acc",
        "val_acc",
    ]:
        assert key in logs.keys()
    for key in logs.keys():
        assert key in [
            "train_parameters",
            "init_parameters",
            "train_loss",
            "val_loss",
            "train_acc",
            "val_acc",
        ]

    for key in ["train_loss", "val_loss", "train_acc", "val_acc"]:
        data = logs[key]
        assert isinstance(data, list)
        assert len(data) == 10

    train_params = logs["train_parameters"]
    assert train_params["train1"] == 2.5
    assert train_params["train2"] == "train"
    init_params = logs["init_parameters"]
    assert init_params["init1"] == 1
    assert init_params["init2"] == "testing"

    shutil.rmtree("models/temp/test_logging", ignore_errors=True)
