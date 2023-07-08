import os.path
import shutil

import numpy as np
import pytest

from src.classification.plant import PlantMFCCCNNClassifier
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


def test_cnn_initialization():
    classifier = PlantMFCCCNNClassifier()
    assert classifier.name == "plant_mfcc_cnn"
    assert not classifier.is_trained

    classifier.data_reader = PlantExperimentDataReader(
        folder=os.path.join("tests", "test_data", "plant")
    )
    with pytest.raises(RuntimeError):
        # Error because not trained
        classifier.classify()


def test_cnn_workflow():
    classifier = PlantMFCCCNNClassifier()
    classifier.data_reader = PlantExperimentDataReader(
        folder=os.path.join("tests", "test_data", "plant")
    )
    assert not classifier.model
    train_parameters = {
        "epochs": 1,
        "which_set": Set.VAL,
        "batch_size": 8,
        "shuffle": False,
        "window": 10,
        "hop": 5,
        "conv_layers": 1,
        "conv_filters": 8,
    }
    classifier.train(train_parameters)
    shutil.rmtree(
        os.path.join("tests", "temp", "plant_mfcc_cnn"), ignore_errors=True
    )
    save_parameters = {
        "save_path": os.path.join("tests", "temp", "plant_mfcc_cnn")
    }
    classifier.save(save_parameters)
    assert os.path.exists(
        os.path.join("tests", "temp", "plant_mfcc_cnn", "saved_model.pb")
    )
    train_parameters["which_set"] = Set.TEST
    results = classifier.classify(train_parameters)
    assert isinstance(results, np.ndarray)
    assert results.shape == (27,)

    new_classifier = PlantMFCCCNNClassifier()
    new_classifier.load(save_parameters)
    new_classifier.data_reader = PlantExperimentDataReader(
        folder=os.path.join("tests", "test_data", "plant")
    )
    new_results = new_classifier.classify(train_parameters)
    assert np.array_equal(new_results, results)

    with pytest.raises(RuntimeError):
        new_classifier.save(
            {"save_path": os.path.join("tests", "temp", "plant_mfcc_cnn")}
        )

    shutil.rmtree(os.path.join("tests", "temp"), ignore_errors=True)
