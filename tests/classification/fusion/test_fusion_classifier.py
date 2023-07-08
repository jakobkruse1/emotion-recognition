import os.path
import shutil

import numpy as np
import pytest

from src.classification.fusion import FusionClassifier
from src.data.fusion_data_reader import FusionProbDataReader, Set


@pytest.mark.filterwarnings("ignore:Data is missing:UserWarning")
def test_fusion_classifier_initialization():
    classifier = FusionClassifier()
    assert not classifier.model
    assert not classifier.is_trained

    classifier.data_reader = FusionProbDataReader(
        folder=os.path.join("tests", "test_data", "fusion")
    )
    with pytest.raises(RuntimeError):
        # Error because not trained
        classifier.classify()


@pytest.mark.filterwarnings("ignore:Data is missing:UserWarning")
def test_fusion_classifier_workflow():
    classifier = FusionClassifier()
    assert not classifier.model
    train_parameters = {
        "epochs": 1,
        "which_set": Set.TRAIN,
        "input_elements": 21,
    }
    classifier.data_reader = FusionProbDataReader(
        folder=os.path.join("tests", "test_data", "fusion")
    )
    classifier.train(train_parameters)
    assert len(classifier.model.layers) == 4

    shutil.rmtree(os.path.join("tests", "temp", "fusion"), ignore_errors=True)
    save_parameters = {"save_path": os.path.join("tests", "temp", "fusion")}
    classifier.save(save_parameters)
    assert os.path.exists(os.path.join("tests", "temp", "fusion"))
    assert os.path.exists(
        os.path.join("tests", "temp", "fusion", "saved_model.pb")
    )
    results = classifier.classify()
    assert isinstance(results, np.ndarray)
    assert results.shape == (123,)

    new_classifier = FusionClassifier()
    new_classifier.load(save_parameters)
    new_classifier.data_reader = FusionProbDataReader(
        folder=os.path.join("tests", "test_data", "fusion")
    )
    new_results = new_classifier.classify()
    assert np.array_equal(new_results, results)

    with pytest.raises(RuntimeError):
        new_classifier.save(
            {"save_path": os.path.join("tests", "temp", "fusion")}
        )

    shutil.rmtree(os.path.join("tests", "temp"), ignore_errors=True)


def test_parameters():
    classifier = FusionClassifier()
    assert not classifier.model
    train_parameters = {
        "epochs": 0,
        "which_set": Set.TRAIN,
    }
    classifier.initialize_model(train_parameters)
    assert len(classifier.model.layers) == 4
    assert vars(classifier.model.layers[2])["bias"].shape[0] == 1024

    train_parameters = {"epochs": 0, "which_set": Set.TRAIN, "hidden_size": 64}
    classifier.initialize_model(train_parameters)
    assert len(classifier.model.layers) == 4
    assert vars(classifier.model.layers[2])["bias"].shape[0] == 64
