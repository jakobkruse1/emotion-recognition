import os.path
import shutil

import numpy as np
import pytest

from src.classification.text import DistilBertClassifier
from src.data.text_data_reader import Set, TextDataReader


def test_distilbert_initialization():
    classifier = DistilBertClassifier()
    assert not classifier.classifier
    assert not classifier.is_trained

    classifier = DistilBertClassifier({"model_name": "bert_123_test"})
    assert classifier.model_name == "distilbert_en_uncased_L-6_H-768_A-12"

    classifier.data_reader = TextDataReader(folder="tests/test_data")
    with pytest.raises(RuntimeError):
        classifier.classify()


def test_distilbert_workflow():
    classifier = DistilBertClassifier()
    assert not classifier.classifier
    train_parameters = {
        "epochs": 1,
        "which_set": Set.TEST,
    }
    classifier.data_reader = TextDataReader(folder="tests/test_data")
    classifier.train(train_parameters)

    shutil.rmtree("tests/temp/bert", ignore_errors=True)
    save_parameters = {"save_path": "tests/temp/bert"}
    classifier.save(save_parameters)
    assert os.path.exists("tests/temp/bert")
    assert os.path.exists("tests/temp/bert/saved_model.pb")
    results = classifier.classify()
    assert isinstance(results, np.ndarray)
    assert results.shape == (30,)

    new_classifier = DistilBertClassifier()
    new_classifier.load(save_parameters)
    new_classifier.data_reader = TextDataReader(folder="tests/test_data")
    new_results = new_classifier.classify()
    assert np.array_equal(new_results, results)

    with pytest.raises(RuntimeError):
        new_classifier.save({"save_path": "tests/temp/bert"})

    shutil.rmtree("tests/temp", ignore_errors=True)
