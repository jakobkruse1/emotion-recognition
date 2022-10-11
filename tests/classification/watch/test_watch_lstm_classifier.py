import os.path
import shutil

import numpy as np
import pytest
import tensorflow as tf

from src.classification.watch import WatchLSTMClassifier
from src.data.watch_exp_reader import Set, WatchExperimentDataReader

CLASS_NAMES = [
    "angry",
    "surprise",
    "disgust",
    "happy",
    "fear",
    "sad",
    "neutral",
]


@pytest.mark.filterwarnings("ignore:Happimeter data:UserWarning")
def test_lstm_initialization():
    classifier = WatchLSTMClassifier()
    assert classifier.name == "watch_lstm"
    assert not classifier.is_trained

    classifier.data_reader = WatchExperimentDataReader(
        folder="tests/test_data/watch"
    )
    with pytest.raises(RuntimeError):
        # Error because not trained
        classifier.classify()


@pytest.mark.filterwarnings("ignore:Happimeter data:UserWarning")
def test_lstm_workflow():
    classifier = WatchLSTMClassifier()
    classifier.data_reader = WatchExperimentDataReader(
        folder="tests/test_data/watch"
    )
    assert not classifier.model
    train_parameters = {
        "epochs": 1,
        "which_set": Set.VAL,
        "batch_size": 8,
        "shuffle": False,
        "window": 10,
        "hop": 5,
        "lstm_layers": 3,
    }
    classifier.train(train_parameters)
    shutil.rmtree("tests/temp/watch_lstm", ignore_errors=True)
    save_parameters = {"save_path": "tests/temp/watch_lstm"}
    classifier.save(save_parameters)
    assert os.path.exists("tests/temp/watch_lstm/saved_model.pb")
    train_parameters["which_set"] = Set.TEST
    results = classifier.classify(train_parameters)
    assert isinstance(results, np.ndarray)
    assert results.shape == (24,)

    new_classifier = WatchLSTMClassifier()
    new_classifier.load(save_parameters)
    new_classifier.data_reader = WatchExperimentDataReader(
        folder="tests/test_data/watch"
    )
    new_results = new_classifier.classify(train_parameters)
    assert np.array_equal(new_results, results)
    lstm_count = 0
    for layer in new_classifier.model.layers:
        if isinstance(layer, tf.keras.layers.Bidirectional):
            lstm_count += 1
    assert lstm_count == 3

    with pytest.raises(RuntimeError):
        new_classifier.save({"save_path": "tests/temp/watch_lstm"})

    shutil.rmtree("tests/temp", ignore_errors=True)


def test_one_lstm_layer():
    classifier = WatchLSTMClassifier()
    classifier.initialize_model({"lstm_layers": 1})
    lstm_count = 0
    for layer in classifier.model.layers:
        print(type(layer))
        if isinstance(layer, tf.keras.layers.Bidirectional):
            lstm_count += 1
    assert lstm_count == 1
