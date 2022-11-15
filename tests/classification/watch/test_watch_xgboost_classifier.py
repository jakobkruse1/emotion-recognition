import os.path
import shutil

import numpy as np
import pytest

from src.classification.watch import WatchXGBoostClassifier
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
def test_xgboost_initialization():
    classifier = WatchXGBoostClassifier()
    assert classifier.name == "xgboost"
    assert not classifier.is_trained

    classifier.data_reader = WatchExperimentDataReader(
        folder="tests/test_data/watch"
    )
    with pytest.raises(RuntimeError):
        # Error because not trained
        classifier.classify()


@pytest.mark.filterwarnings("ignore:Happimeter data:UserWarning")
def test_xgboost_workflow():
    classifier = WatchXGBoostClassifier()
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
    shutil.rmtree("tests/temp/xgboost", ignore_errors=True)
    save_parameters = {"save_path": "tests/temp/xgboost"}
    classifier.save(save_parameters)
    assert os.path.exists("tests/temp/xgboost/model.bin")
    train_parameters["which_set"] = Set.TEST
    results = classifier.classify(train_parameters)
    assert isinstance(results, np.ndarray)
    assert results.shape == (24,)

    new_classifier = WatchXGBoostClassifier()
    new_classifier.load(save_parameters)
    new_classifier.data_reader = WatchExperimentDataReader(
        folder="tests/test_data/watch"
    )
    new_results = new_classifier.classify(train_parameters)
    assert np.array_equal(new_results, results)

    with pytest.raises(RuntimeError):
        new_classifier.save({"save_path": "tests/temp/xgboost"})

    shutil.rmtree("tests/temp", ignore_errors=True)
