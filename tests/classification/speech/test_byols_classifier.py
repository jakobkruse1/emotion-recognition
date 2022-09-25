import os.path
import shutil

import numpy as np
import pytest

from src.classification.speech import BYOLSClassifier
from src.data.speech_data_reader import Set, SpeechDataReader


def test_byols_initialization():
    classifier = BYOLSClassifier()
    assert not classifier.model
    assert not classifier.is_trained

    classifier.data_reader = SpeechDataReader(folder="tests/test_data/speech")
    with pytest.raises(RuntimeError):
        # Error because not trained
        classifier.classify()


def test_byols_workflow():
    classifier = BYOLSClassifier()
    assert not classifier.model
    train_parameters = {
        "epochs": 1,
        "which_set": Set.VAL,
        "batch_size": 8,
        "max_elements": 7,
        "shuffle": False,
        "freeze": True,
    }
    classifier.data_reader = SpeechDataReader(folder="tests/test_data/speech")
    classifier.train(train_parameters)

    shutil.rmtree("tests/temp/byols", ignore_errors=True)
    save_parameters = {"save_path": "tests/temp/byols"}
    classifier.save(save_parameters)
    assert os.path.exists("tests/temp/byols/byols.pth")
    assert os.path.exists("tests/temp/byols/model.txt")
    results = classifier.classify({"max_elements": 7, "shuffle": False})
    assert isinstance(results, np.ndarray)
    assert results.shape == (7,)

    new_classifier = BYOLSClassifier()
    new_classifier.load(save_parameters)
    new_classifier.data_reader = SpeechDataReader(
        folder="tests/test_data/speech"
    )
    new_results = new_classifier.classify(
        {"max_elements": 7, "shuffle": False}
    )
    assert np.array_equal(new_results, results)

    with pytest.raises(RuntimeError):
        new_classifier.save({"save_path": "tests/temp/byols/byols.pth"})

    shutil.rmtree("tests/temp", ignore_errors=True)
