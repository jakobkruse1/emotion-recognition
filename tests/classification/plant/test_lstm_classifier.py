import os.path
import shutil

import numpy as np
import pytest
import tensorflow as tf

from src.classification.plant import PlantLSTMClassifier
from src.data.plant_exp_reader import PlantExperimentDataReader, Set

CLASS_NAMES = [
    "angry",
    "surprise",
    "disgust",
    "happy",
    "fear",
    "sad",
    "neutral",
]


def test_lstm_initialization():
    classifier = PlantLSTMClassifier()
    assert classifier.name == "plant_lstm"
    assert not classifier.is_trained

    classifier.data_reader = PlantExperimentDataReader(
        folder="tests/test_data/plant"
    )
    with pytest.raises(RuntimeError):
        # Error because not trained
        classifier.classify()


def test_lstm_workflow():
    classifier = PlantLSTMClassifier()
    classifier.data_reader = PlantExperimentDataReader(
        folder="tests/test_data/plant"
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
    shutil.rmtree("tests/temp/plant_lstm", ignore_errors=True)
    save_parameters = {"save_path": "tests/temp/plant_lstm"}
    classifier.save(save_parameters)
    assert os.path.exists("tests/temp/plant_lstm/saved_model.pb")
    train_parameters["which_set"] = Set.TEST
    results = classifier.classify(train_parameters)
    assert isinstance(results, np.ndarray)
    assert results.shape == (27,)

    new_classifier = PlantLSTMClassifier()
    new_classifier.load(save_parameters)
    new_classifier.data_reader = PlantExperimentDataReader(
        folder="tests/test_data/plant"
    )
    new_results = new_classifier.classify(train_parameters)
    assert np.array_equal(new_results, results)
    lstm_count = 0
    for layer in new_classifier.model.layers:
        if isinstance(layer, tf.keras.layers.Bidirectional):
            lstm_count += 1
    assert lstm_count == 3

    with pytest.raises(RuntimeError):
        new_classifier.save({"save_path": "tests/temp/plant_lstm"})

    shutil.rmtree("tests/temp", ignore_errors=True)


def test_one_lstm_layer():
    classifier = PlantLSTMClassifier()
    classifier.initialize_model({"lstm_layers": 1})
    lstm_count = 0
    for layer in classifier.model.layers:
        print(type(layer))
        if isinstance(layer, tf.keras.layers.Bidirectional):
            lstm_count += 1
    assert lstm_count == 1
