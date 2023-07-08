""" Test base functionality in the plant emotion classifier. """
import os
from typing import Dict

import numpy as np
import tensorflow as tf

from src.classification.plant import PlantEmotionClassifier
from src.data.balanced_plant_exp_reader import (
    BalancedPlantExperimentDataReader,
)
from src.data.data_reader import Set
from src.data.plant_exp_reader import PlantExperimentDataReader


class TestClassifier(PlantEmotionClassifier):
    def train(self, parameters: Dict = None, **kwargs) -> None:
        pass

    def load(self, parameters: Dict = None, **kwargs) -> None:
        pass

    def save(self, parameters: Dict = None, **kwargs) -> None:
        pass

    def classify(self, parameters: Dict = None, **kwargs) -> np.array:
        pass


def test_init():
    classifier = TestClassifier()
    assert classifier.name == "plant"
    assert classifier.data_type == "plant"
    assert not classifier.is_trained
    members = [
        "callbacks",
        "optimizer",
        "loss",
        "metrics",
        "train_data",
        "val_data",
        "class_weights",
    ]
    for member in members:
        assert classifier.__getattribute__(member) is None


def test_prepare_training():
    classifier = TestClassifier()
    classifier.prepare_training({"learning_rate": 4, "patience": 169})
    members = ["callbacks", "optimizer", "loss", "metrics"]
    for member in members:
        assert classifier.__getattribute__(member) is not None
    assert classifier.optimizer.lr == 4
    assert classifier.callbacks[0].patience == 169
    assert len(classifier.callbacks) == 1

    classifier.prepare_training(
        {"learning_rate": 4, "patience": 169, "checkpoint": True}
    )
    assert len(classifier.callbacks) == 2


def test_prepare_data():
    classifier = TestClassifier()
    classifier.data_reader = PlantExperimentDataReader(
        folder=os.path.join("tests", "test_data", "plant")
    )
    classifier.prepare_data({"which_set": Set.VAL, "batch_size": 8})
    members = ["train_data", "val_data"]
    for member in members:
        assert classifier.__getattribute__(member) is not None
    assert classifier.class_weights is None


def test_weights():
    # Only testing to switch weights on/off. Correctness is checked elsewhere.
    classifier = TestClassifier()
    classifier.data_reader = PlantExperimentDataReader(
        folder=os.path.join("tests", "test_data", "plant")
    )
    classifier.prepare_data(
        {"which_set": Set.VAL, "batch_size": 8, "weighted": True}
    )
    members = ["train_data", "val_data", "class_weights"]
    for member in members:
        assert classifier.__getattribute__(member) is not None


def test_balancing():
    classifier = TestClassifier()
    classifier.data_reader = PlantExperimentDataReader(
        folder=os.path.join("tests", "test_data", "plant")
    )
    classifier.prepare_data(
        {"which_set": Set.VAL, "batch_size": 8, "balanced": True}
    )
    members = ["train_data", "val_data"]
    for member in members:
        assert classifier.__getattribute__(member) is not None
    assert classifier.class_weights is None
    assert isinstance(
        classifier.data_reader, BalancedPlantExperimentDataReader
    )
    assert classifier.data_reader.folder == os.path.join(
        "tests", "test_data", "plant"
    )


def test_mfcc():
    input = np.random.randn(1, 10000)
    assert input.shape == (1, 10000)
    input_tensor = tf.convert_to_tensor(input, dtype=tf.float32)
    mfcc = TestClassifier.compute_mfccs(input_tensor, {})
    assert mfcc.shape == (1, 7, 20)
    mfcc = TestClassifier.compute_mfccs(input_tensor, {"num_mfcc": 60})
    assert mfcc.shape == (1, 7, 60)
