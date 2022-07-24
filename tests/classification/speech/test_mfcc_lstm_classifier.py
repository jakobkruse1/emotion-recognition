import os.path
import shutil

import numpy as np
import pytest

from src.classification.speech import MFCCLSTMClassifier
from src.data.speech_data_reader import Set, SpeechDataReader


def test_mfcc_lstm_initialization():
    classifier = MFCCLSTMClassifier()
    assert not classifier.model
    assert not classifier.is_trained

    classifier.data_reader = SpeechDataReader(folder="tests/test_data/speech")
    with pytest.raises(RuntimeError):
        # Error because not trained
        classifier.classify()


def test_mfcc_lstm_workflow():
    classifier = MFCCLSTMClassifier()
    assert not classifier.model
    train_parameters = {
        "epochs": 1,
        "which_set": Set.VAL,
        "batch_size": 8,
        "max_elements": 7,
        "shuffle": False,
    }
    classifier.data_reader = SpeechDataReader(folder="tests/test_data/speech")
    classifier.train(train_parameters)

    shutil.rmtree("tests/temp/mfcc_lstm", ignore_errors=True)
    save_parameters = {"save_path": "tests/temp/mfcc_lstm"}
    classifier.save(save_parameters)
    assert os.path.exists("tests/temp/mfcc_lstm/saved_model.pb")
    assert os.path.exists("tests/temp/mfcc_lstm/keras_metadata.pb")
    results = classifier.classify({"max_elements": 7, "shuffle": False})
    assert isinstance(results, np.ndarray)
    assert results.shape == (7,)

    new_classifier = MFCCLSTMClassifier()
    new_classifier.load(save_parameters)
    new_classifier.data_reader = SpeechDataReader(
        folder="tests/test_data/speech"
    )
    new_results = new_classifier.classify(
        {"max_elements": 7, "shuffle": False}
    )
    assert np.array_equal(new_results, results)

    with pytest.raises(RuntimeError):
        new_classifier.save({"save_path": "tests/temp/mfcc_lstm"})

    shutil.rmtree("tests/temp", ignore_errors=True)
