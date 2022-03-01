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

    with pytest.raises(RuntimeError):
        classifier.classify()


def test_bert_workflow():
    classifier = BertClassifier(
        {"model_name": "bert_en_uncased_L-2_H-128_A-2"}
    )
    assert not classifier.classifier
    train_parameters = {
        "epochs": 5,
        "set": Set.TEST,  # Speedup training by using test set which is smaller
    }
    classifier.data_reader = TextDataReader(folder="tests/test_data")
    classifier.data_reader.file_map[Set.TEST] = "text_test.csv"
    classifier.data_reader.file_map[Set.VAL] = "text_test.csv"
    classifier.train(train_parameters)

    shutil.rmtree("tests/temp/bert", ignore_errors=True)
    save_parameters = {"save_path": "tests/temp/bert"}
    classifier.save(save_parameters)
    assert os.path.exists("tests/temp/bert")
    assert os.path.exists("tests/temp/bert/saved_model.pb")
    results = classifier.classify()
    assert isinstance(results, np.ndarray)
    assert results.shape == (30,)

    new_classifier = BertClassifier(
        {"model_name": "bert_en_uncased_L-2_H-128_A-2"}
    )
    new_classifier.load(save_parameters)
    new_classifier.data_reader = TextDataReader(folder="tests/test_data")
    new_classifier.data_reader.file_map[Set.TEST] = "text_test.csv"
    new_results = new_classifier.classify()
    assert np.array_equal(new_results, results)

    with pytest.raises(RuntimeError):
        new_classifier.save({"save_path": "tests/temp/bert"})

    shutil.rmtree("tests/temp/bert", ignore_errors=True)
