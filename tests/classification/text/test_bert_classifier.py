import os.path
import shutil

import numpy as np
import pytest

from src.classification.text import BertClassifier
from src.data.text_data_reader import Set, TextDataReader


def test_bert_initialization():
    classifier = BertClassifier()
    assert not classifier.classifier
    assert not classifier.is_trained

    classifier = BertClassifier({"model_name": "bert_123_test"})
    assert classifier.model_name == "bert_123_test"

    classifier.data_reader = TextDataReader(
        folder=os.path.join("tests", "test_data", "text")
    )
    with pytest.raises(RuntimeError):
        classifier.classify()


def test_bert_workflow():
    classifier = BertClassifier(
        {"model_name": "bert_en_uncased_L-2_H-128_A-2"}
    )
    assert not classifier.classifier
    train_parameters = {
        "epochs": 5,
        "dense_layer": 3,
        "which_set": Set.TEST,
    }
    classifier.data_reader = TextDataReader(
        folder=os.path.join("tests", "test_data", "text")
    )
    classifier.train(train_parameters)

    shutil.rmtree(os.path.join("tests", "temp", "bert"), ignore_errors=True)
    save_parameters = {"save_path": os.path.join("tests", "temp", "bert")}
    classifier.save(save_parameters)
    assert os.path.exists(os.path.join("tests", "temp", "bert"))
    assert os.path.exists(
        os.path.join("tests", "temp", "bert", "saved_model.pb")
    )
    results = classifier.classify()
    assert isinstance(results, np.ndarray)
    assert results.shape == (30,)

    new_classifier = BertClassifier(
        {"model_name": "bert_en_uncased_L-2_H-128_A-2"}
    )
    new_classifier.load(save_parameters)
    new_classifier.data_reader = TextDataReader(
        folder=os.path.join("tests", "test_data", "text")
    )
    new_results = new_classifier.classify()
    assert np.array_equal(new_results, results)

    with pytest.raises(RuntimeError):
        new_classifier.save(
            {"save_path": os.path.join("tests", "temp", "bert")}
        )

    shutil.rmtree(os.path.join("tests", "temp"), ignore_errors=True)
