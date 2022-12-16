""" Test the fusion data reader. """
import numpy as np
import pytest
import tensorflow as tf

from src.data.fusion_data_reader import FusionProbDataReader, Set


def test_fusion_initialization():
    dr = FusionProbDataReader()
    assert dr.name == "fusion"
    assert dr.folder == "data/continuous"
    assert dr.default_label_mode == "expected"
    for mod in ["image", "plant", "watch"]:
        assert dr.feature_sizes[mod] == 7
    dr.cleanup()


@pytest.mark.filterwarnings("ignore:Data is missing:UserWarning")
def test_splitting():
    dr = FusionProbDataReader(folder="tests/test_data/fusion")
    assert dr.folder == "tests/test_data/fusion"
    for which_set in [Set.TRAIN, Set.VAL, Set.TEST, Set.ALL]:
        dataset = dr.get_emotion_data("neutral_ekman", which_set, batch_size=8)
        assert isinstance(dataset, tf.data.Dataset)
        all_data = np.empty((0, 21))
        all_labels = np.empty((0, 7))
        for data, labels in dataset:
            all_data = np.concatenate([all_data, data], axis=0)
            all_labels = np.concatenate([all_labels, labels], axis=0)

        if which_set == Set.TRAIN:
            size = 367
        elif which_set == Set.VAL:
            size = 123
        elif which_set == Set.TEST:
            size = 123
        else:
            size = 613
        assert all_labels.shape == (size, 7)
        assert all_data.shape == (size, 21)


@pytest.mark.filterwarnings("ignore:Data is missing:UserWarning")
def test_data_reading():
    dr = FusionProbDataReader(folder="tests/test_data/fusion")
    dataset = dr.get_emotion_data("neutral_ekman", Set.TEST, batch_size=8)
    batch = 0
    for data, labels in dataset.take(15):
        batch += 1
        assert data.numpy().shape == (8, 21)
        assert labels.numpy().shape == (8, 7)
        for label in labels.numpy():
            assert label.shape == (7,)
            assert np.sum(label) == 1
    assert batch == 15


@pytest.mark.filterwarnings("ignore:Data is missing:UserWarning")
def test_reading_three():
    dr = FusionProbDataReader(folder="tests/test_data/fusion")
    assert dr.folder == "tests/test_data/fusion"
    dataset = dr.get_emotion_data("three", Set.TEST, batch_size=4)
    assert isinstance(dataset, tf.data.Dataset)
    batch = 0
    for texts, labels in dataset.take(15):
        batch += 1
        assert texts.numpy().shape == (4, 21)
        assert labels.numpy().shape == (4, 3)
        for label in labels.numpy():
            assert label.shape == (3,)
            assert np.sum(label) == 1
    assert batch == 15


@pytest.mark.filterwarnings("ignore:Data is missing:UserWarning")
def test_labels():
    dr = FusionProbDataReader(folder="tests/test_data/fusion")
    assert dr.folder == "tests/test_data/fusion"
    dataset = dr.get_emotion_data(
        "neutral_ekman", Set.ALL, batch_size=8, parameters={"shuffle": False}
    )
    dataset_labels = np.empty((0,))
    dataset_data = np.empty((0, 21))
    dataset_raw_labels = np.empty((0, 7))
    for data, labels in dataset:
        dataset_data = np.concatenate([dataset_data, data.numpy()], axis=0)
        labels = labels.numpy()
        dataset_raw_labels = np.concatenate(
            [dataset_raw_labels, labels], axis=0
        )
        labels = np.argmax(labels, axis=1)
        dataset_labels = np.concatenate([dataset_labels, labels], axis=0)
    true_labels = dr.get_labels(Set.ALL)
    assert true_labels.shape == (613,)
    assert dataset_labels.shape == (613,)
    assert np.array_equal(true_labels, dataset_labels)
    d_data, d_labels = FusionProbDataReader.convert_to_numpy(dataset)
    assert np.array_equal(d_data, dataset_data)
    assert np.array_equal(d_labels, dataset_raw_labels)


@pytest.mark.filterwarnings("ignore:Data is missing:UserWarning")
def test_modalities():
    dr = FusionProbDataReader(folder="tests/test_data/fusion")
    assert dr.folder == "tests/test_data/fusion"
    dataset = dr.get_emotion_data(
        "three", Set.TEST, batch_size=4, parameters={"modalities": ["image"]}
    )
    for data, labels in dataset.take(1):
        assert data.shape == (4, 7)
        assert labels.shape == (4, 3)

    dataset = dr.get_emotion_data(
        "neutral_ekman",
        Set.TEST,
        batch_size=4,
        parameters={"modalities": ["plant", "image"]},
    )
    for data, labels in dataset.take(1):
        assert data.shape == (4, 14)
        assert labels.shape == (4, 7)
