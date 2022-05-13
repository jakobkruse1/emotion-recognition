import os.path
import shutil

import numpy as np
import pytest

from src.classification.image import VGG16Classifier
from src.data.image_data_reader import ImageDataReader, Set


def test_vgg16_initialization():
    classifier = VGG16Classifier()
    assert not classifier.model
    assert not classifier.is_trained

    classifier.data_reader = ImageDataReader(folder="tests/test_data/image")
    with pytest.raises(RuntimeError):
        # Error because not trained
        classifier.classify()


def test_efficientnet_workflow():
    classifier = VGG16Classifier()
    assert not classifier.model
    train_parameters = {"epochs": 1, "which_set": Set.TRAIN, "deep": False}
    classifier.data_reader = ImageDataReader(folder="tests/test_data/image")
    classifier.train(train_parameters)
    assert len(classifier.model.layers) == 5

    shutil.rmtree("tests/temp/vgg", ignore_errors=True)
    save_parameters = {"save_path": "tests/temp/vgg"}
    classifier.save(save_parameters)
    assert os.path.exists("tests/temp/vgg")
    assert os.path.exists("tests/temp/vgg/saved_model.pb")
    results = classifier.classify()
    assert isinstance(results, np.ndarray)
    assert results.shape == (7,)

    new_classifier = VGG16Classifier()
    new_classifier.load(save_parameters)
    new_classifier.data_reader = ImageDataReader(
        folder="tests/test_data/image"
    )
    new_results = new_classifier.classify()
    assert np.array_equal(new_results, results)

    with pytest.raises(RuntimeError):
        new_classifier.save({"save_path": "tests/temp/vgg"})

    shutil.rmtree("tests/temp", ignore_errors=True)


def test_parameters():
    classifier = VGG16Classifier()
    assert not classifier.model
    train_parameters = {
        "epochs": 0,
        "which_set": Set.TRAIN,
        "deep": False,
        "dropout": 0,
    }
    classifier.initialize_model(train_parameters)
    assert len(classifier.model.layers) == 5

    train_parameters = {
        "epochs": 0,
        "which_set": Set.TRAIN,
        "deep": True,
        "dropout": 0,
    }
    classifier.initialize_model(train_parameters)
    assert len(classifier.model.layers) == 7

    train_parameters = {
        "epochs": 0,
        "which_set": Set.TRAIN,
        "deep": True,
        "dropout": 0.2,
    }
    classifier.initialize_model(train_parameters)
    assert len(classifier.model.layers) == 10
