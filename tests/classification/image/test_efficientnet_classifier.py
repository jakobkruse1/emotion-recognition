import os.path
import shutil

import numpy as np
import pytest

from src.classification.image import MultiTaskEfficientNetB2Classifier
from src.data.image_data_reader import ImageDataReader, Set


def test_efficientnet_initialization():
    classifier = MultiTaskEfficientNetB2Classifier()
    assert not classifier.model
    assert not classifier.is_trained

    classifier.data_reader = ImageDataReader(folder="tests/test_data/image")
    with pytest.raises(RuntimeError):
        # Error because not trained
        classifier.classify()


def test_efficientnet_workflow():
    classifier = MultiTaskEfficientNetB2Classifier()
    assert not classifier.model
    train_parameters = {
        "epochs": 1,
        "which_set": Set.TRAIN,
    }
    classifier.data_reader = ImageDataReader(folder="tests/test_data/image")
    print(1)
    classifier.train(train_parameters)
    print(2)
    assert len(classifier.model.layers) == 4
    print(3)
    shutil.rmtree("tests/temp/efficient", ignore_errors=True)
    print(4)
    save_parameters = {"save_path": "tests/temp/efficient"}
    classifier.save(save_parameters)
    print(5)
    assert os.path.exists("tests/temp/efficient")
    assert os.path.exists("tests/temp/efficient/saved_model.pb")
    print(6)
    results = classifier.classify()
    print(7)
    assert isinstance(results, np.ndarray)
    assert results.shape == (7,)
    print(8)
    new_classifier = MultiTaskEfficientNetB2Classifier()
    print(9)
    new_classifier.load(save_parameters)
    print(10)
    new_classifier.data_reader = ImageDataReader(
        folder="tests/test_data/image"
    )
    print(11)
    new_results = new_classifier.classify()
    print(12)
    assert np.array_equal(new_results, results)

    with pytest.raises(RuntimeError):
        new_classifier.save({"save_path": "tests/temp/efficient"})
    print(13)
    shutil.rmtree("tests/temp", ignore_errors=True)


def test_extra_layer():
    classifier = MultiTaskEfficientNetB2Classifier()
    assert not classifier.model
    train_parameters = {
        "epochs": 1,
        "which_set": Set.TRAIN,
        "extra_layer": 1024,
    }
    classifier.initialize_model(train_parameters)
    assert len(classifier.model.layers) == 5
