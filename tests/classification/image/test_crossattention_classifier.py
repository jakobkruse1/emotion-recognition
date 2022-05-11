import os.path
import shutil

import numpy as np
import pytest

from src.classification.image import CrossAttentionNetworkClassifier
from src.data.image_data_reader import ImageDataReader, Set


def test_crossattention_initialization():
    classifier = CrossAttentionNetworkClassifier()
    assert not classifier.model
    assert not classifier.is_trained

    classifier.data_reader = ImageDataReader(folder="tests/test_data/image")
    with pytest.raises(RuntimeError):
        # Error because not trained
        classifier.classify()


def test_crossattention_workflow():
    classifier = CrossAttentionNetworkClassifier()
    assert not classifier.model
    train_parameters = {
        "epochs": 2,
        "which_set": Set.TRAIN,
        "gpu": 1,  # To see if this is caught
    }
    classifier.data_reader = ImageDataReader(folder="tests/test_data/image")
    classifier.train(train_parameters)

    shutil.rmtree("tests/temp/cross_attention", ignore_errors=True)
    save_parameters = {"save_path": "tests/temp/cross_attention/ca.pth"}
    classifier.save(save_parameters)
    assert os.path.exists("tests/temp/cross_attention/ca.pth")
    assert os.path.exists("tests/temp/cross_attention/ca.pth")
    results = classifier.classify()
    assert isinstance(results, np.ndarray)
    assert results.shape == (7,)

    new_classifier = CrossAttentionNetworkClassifier()
    new_classifier.load(save_parameters)
    new_classifier.data_reader = ImageDataReader(
        folder="tests/test_data/image"
    )
    new_results = new_classifier.classify()
    assert np.array_equal(new_results, results)

    with pytest.raises(RuntimeError):
        new_classifier.save({"save_path": "tests/temp/cross_attention/ca.pth"})

    shutil.rmtree("tests/temp", ignore_errors=True)
