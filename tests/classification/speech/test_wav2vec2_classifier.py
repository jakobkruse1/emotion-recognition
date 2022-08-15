import os.path
import shutil

import numpy as np
import pytest

from src.classification.speech import Wav2Vec2Classifier
from src.data.speech_data_reader import Set, SpeechDataReader


def test_wav2vec2_initialization():
    classifier = Wav2Vec2Classifier()
    assert not classifier.model
    assert not classifier.is_trained

    classifier.data_reader = SpeechDataReader(folder="tests/test_data/speech")
    with pytest.raises(RuntimeError):
        # Error because not trained
        classifier.classify()


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_wav2vec2_workflow():
    classifier = Wav2Vec2Classifier()
    assert not classifier.model
    train_parameters = {
        "epochs": 1,
        "which_set": Set.VAL,
        "batch_size": 8,
        "max_elements": 7,
        "shuffle": False,
        "freeze": True,
        "extra_layer": 1024,
    }
    classifier.data_reader = SpeechDataReader(folder="tests/test_data/speech")
    classifier.train(train_parameters)

    assert classifier.model.hidden is not None

    shutil.rmtree("tests/temp/wav2vec2", ignore_errors=True)
    save_parameters = {
        "save_path": "tests/temp/wav2vec2/wav2vec2.pth",
        "extra_layer": 1024,
    }
    classifier.save(save_parameters)
    assert os.path.exists("tests/temp/wav2vec2/wav2vec2.pth")
    results = classifier.classify({"max_elements": 7, "shuffle": False})
    assert isinstance(results, np.ndarray)
    assert results.shape == (7,)

    new_classifier = Wav2Vec2Classifier()
    new_classifier.load(save_parameters)
    new_classifier.data_reader = SpeechDataReader(
        folder="tests/test_data/speech"
    )
    new_results = new_classifier.classify(
        {"max_elements": 7, "shuffle": False}
    )
    assert np.array_equal(new_results, results)

    with pytest.raises(RuntimeError):
        new_classifier.save({"save_path": "tests/temp/wav2vec2/wav2vec2.pth"})

    shutil.rmtree("tests/temp", ignore_errors=True)


def test_no_extra_layer():
    classifier = Wav2Vec2Classifier()
    assert not classifier.model
    classifier.data_reader = SpeechDataReader(folder="tests/test_data/speech")
    train_parameters = {
        "epochs": 0,
        "which_set": Set.VAL,
        "batch_size": 8,
        "max_elements": 7,
        "freeze": True,
        "shuffle": False,
        "dataset": "meld",
    }
    classifier.train(train_parameters)
    assert classifier.model.hidden is None
