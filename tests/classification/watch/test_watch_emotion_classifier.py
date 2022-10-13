""" Test base functionality in the watch emotion classifier. """
from typing import Dict

import numpy as np
import pytest

from src.classification.watch import WatchEmotionClassifier
from src.data.balanced_watch_exp_reader import (
    BalancedWatchExperimentDataReader,
)
from src.data.data_reader import Set
from src.data.watch_exp_reader import WatchExperimentDataReader


class TestClassifier(WatchEmotionClassifier):
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
    assert classifier.name == "watch"
    assert classifier.data_type == "watch"
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


@pytest.mark.filterwarnings("ignore:Happimeter data:UserWarning")
def test_prepare_data():
    classifier = TestClassifier()
    classifier.data_reader = WatchExperimentDataReader(
        folder="tests/test_data/watch"
    )
    classifier.prepare_data({"which_set": Set.VAL, "batch_size": 8})
    members = ["train_data", "val_data"]
    for member in members:
        assert classifier.__getattribute__(member) is not None
    assert classifier.class_weights is None


@pytest.mark.filterwarnings("ignore:Happimeter data:UserWarning")
def test_weights():
    # Only testing to switch weights on/off. Correctness is checked elsewhere.
    classifier = TestClassifier()
    classifier.data_reader = WatchExperimentDataReader(
        folder="tests/test_data/watch"
    )
    classifier.prepare_data(
        {"which_set": Set.VAL, "batch_size": 8, "weighted": True}
    )
    members = ["train_data", "val_data", "class_weights"]
    for member in members:
        assert classifier.__getattribute__(member) is not None


@pytest.mark.filterwarnings("ignore:Happimeter data:UserWarning")
def test_balancing():
    classifier = TestClassifier()
    classifier.data_reader = WatchExperimentDataReader(
        folder="tests/test_data/watch"
    )
    classifier.prepare_data(
        {"which_set": Set.VAL, "batch_size": 8, "balanced": True}
    )
    members = ["train_data", "val_data"]
    for member in members:
        assert classifier.__getattribute__(member) is not None
    assert classifier.class_weights is None
    assert isinstance(
        classifier.data_reader, BalancedWatchExperimentDataReader
    )
    assert classifier.data_reader.folder == "tests/test_data/watch"
