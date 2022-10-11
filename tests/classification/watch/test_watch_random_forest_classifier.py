import os.path
import shutil

import numpy as np
import pytest

from src.classification.watch import WatchRandomForestClassifier
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
def test_random_forest_initialization():
    classifier = WatchRandomForestClassifier()
    assert classifier.name == "random_forest"
    assert not classifier.is_trained

    classifier.data_reader = WatchExperimentDataReader(
        folder="tests/test_data/watch"
    )
    with pytest.raises(RuntimeError):
        # Error because not trained
        classifier.classify()


@pytest.mark.filterwarnings("ignore:Happimeter data:UserWarning")
def test_random_forest_workflow():
    classifier = WatchRandomForestClassifier()
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
    }
    classifier.train(train_parameters)
    shutil.rmtree("tests/temp/random_forest", ignore_errors=True)
    save_parameters = {"save_path": "tests/temp/random_forest"}
    classifier.save(save_parameters)
    assert os.path.exists("tests/temp/random_forest/model.pkl")
    train_parameters["which_set"] = Set.TEST
    results = classifier.classify(train_parameters)
    assert isinstance(results, np.ndarray)
    assert results.shape == (24,)

    new_classifier = WatchRandomForestClassifier()
    new_classifier.load(save_parameters)
    new_classifier.data_reader = WatchExperimentDataReader(
        folder="tests/test_data/watch"
    )
    new_results = new_classifier.classify(train_parameters)
    assert np.array_equal(new_results, results)

    with pytest.raises(RuntimeError):
        new_classifier.save({"save_path": "tests/temp/random_forest"})

    shutil.rmtree("tests/temp", ignore_errors=True)
