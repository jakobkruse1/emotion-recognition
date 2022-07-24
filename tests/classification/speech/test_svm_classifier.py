import os.path
import shutil

import numpy as np
import pytest

from src.classification.speech import SVMClassifier
from src.data.speech_data_reader import Set, SpeechDataReader


def test_svm_initialization():
    classifier = SVMClassifier()
    assert not classifier.model
    assert not classifier.scaler
    assert not classifier.is_trained

    classifier.data_reader = SpeechDataReader(folder="tests/test_data/speech")
    with pytest.raises(RuntimeError):
        # Error because not trained
        classifier.classify()


def test_svm_workflow():
    classifier = SVMClassifier()
    train_parameters = {
        "epochs": 1,
        "which_set": Set.VAL,
        "batch_size": 8,
        "shuffle": False,
        "dataset": "meld",
    }
    classifier.data_reader = SpeechDataReader(folder="tests/test_data/speech")
    classifier.train(train_parameters)

    shutil.rmtree("tests/temp/svm", ignore_errors=True)
    save_parameters = {"save_path": "tests/temp/svm"}
    classifier.save(save_parameters)
    assert os.path.exists("tests/temp/svm/model.pkl")
    assert os.path.exists("tests/temp/svm/scaler.pkl")
    results = classifier.classify({"shuffle": False, "dataset": "meld"})
    assert isinstance(results, np.ndarray)
    assert results.shape == (7,)

    new_classifier = SVMClassifier()
    new_classifier.load(save_parameters)
    new_classifier.data_reader = SpeechDataReader(
        folder="tests/test_data/speech"
    )
    new_results = new_classifier.classify(
        {"shuffle": False, "dataset": "meld"}
    )
    assert np.array_equal(new_results, results)

    new_classifier = SVMClassifier()
    with pytest.raises(RuntimeError):
        new_classifier.save({"save_path": "tests/temp/svm"})

    shutil.rmtree("tests/temp", ignore_errors=True)
