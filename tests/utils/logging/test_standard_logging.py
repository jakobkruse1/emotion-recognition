""" Test the standard logger. """

import json
import os.path
import shutil
from typing import Any, Dict

import numpy as np

from src.classification.emotion_classifier import EmotionClassifier
from src.utils import logging


class NoopClassifier(EmotionClassifier):
    def __init__(self, parameters: Dict[str, Any]):
        super().__init__("noop", "text", parameters)
        self.logger = logging.StandardLogger()
        self.logger.log_start({"init_parameters": parameters})

    def train(self, parameters: Dict, **kwargs) -> None:
        self.logger.log_start({"train_parameters": parameters})
        for epoch in range(10):
            self.logger.log_epoch(
                {
                    "train_loss": 10 - epoch,
                    "train_acc": epoch / 2,
                    "val_loss": 12 - epoch,
                    "val_acc": epoch / 3,
                }
            )
        self.logger.log_end({"final": 747})

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
    assert "train_acc" not in classifier.logger.logs.keys()
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
        "final",
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
            "final",
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
    assert logs["final"] == 747

    shutil.rmtree("models/temp/test_logging", ignore_errors=True)
