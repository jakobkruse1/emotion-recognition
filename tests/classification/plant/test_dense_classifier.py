import os.path
import shutil

import numpy as np
import pytest

from src.classification.plant import PlantDenseClassifier
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


def test_dense_initialization():
    classifier = PlantDenseClassifier()
    assert classifier.name == "plant_dense"
    assert not classifier.is_trained

    classifier.data_reader = PlantExperimentDataReader(
        folder="tests/test_data/plant"
    )
    with pytest.raises(RuntimeError):
        # Error because not trained
        classifier.classify()


def test_dense_workflow():
    classifier = PlantDenseClassifier()
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
        "dense_units": 64,
        "hidden_layers": 2,
    }
    classifier.train(train_parameters)
    shutil.rmtree("tests/temp/plant_dense", ignore_errors=True)
    save_parameters = {"save_path": "tests/temp/plant_dense"}
    classifier.save(save_parameters)
    assert os.path.exists("tests/temp/plant_dense/saved_model.pb")
    train_parameters["which_set"] = Set.TEST
    results = classifier.classify(train_parameters)
    assert isinstance(results, np.ndarray)
    assert results.shape == (27,)

    new_classifier = PlantDenseClassifier()
    new_classifier.load(save_parameters)
    new_classifier.data_reader = PlantExperimentDataReader(
        folder="tests/test_data/plant"
    )
    new_results = new_classifier.classify(train_parameters)
    assert np.array_equal(new_results, results)

    with pytest.raises(RuntimeError):
        new_classifier.save({"save_path": "tests/temp/plant_dense"})

    shutil.rmtree("tests/temp", ignore_errors=True)
