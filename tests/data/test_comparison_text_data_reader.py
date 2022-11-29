"""Test the comparison text data reader"""

import numpy as np
import pytest
import tensorflow as tf

from src.data.comparison_text_data_reader import ComparisonTextDataReader, Set


def test_initialization():
    dr = ComparisonTextDataReader()
    assert dr.name == "comparison_text"
    assert dr.folder == "data/comparison_dataset/text"
    dr.cleanup()


def test_reading():
    dr = ComparisonTextDataReader(folder="tests/test_data/text/comparison")
    assert dr.folder == "tests/test_data/text/comparison"
    dataset = dr.get_emotion_data("neutral_ekman", Set.TEST, batch_size=8)
    assert isinstance(dataset, tf.data.Dataset)
    batch = 0
    for texts, labels in dataset:
        batch += 1
        assert texts.numpy().shape == (7,)
        assert labels.numpy().shape == (7, 7)
        for text in texts.numpy():
            text = str(text)
            assert len(text) > 5
        for label in labels.numpy():
            assert label.shape == (7,)
            assert np.sum(label) == 1
    assert batch == 1

    with pytest.raises(AssertionError):
        _ = dr.get_seven_emotion_data(Set.TRAIN)

    with pytest.raises(AssertionError):
        _ = dr.get_seven_emotion_data(Set.VAL)

    with pytest.raises(ValueError):
        _ = dr.get_emotion_data("wrong")


def test_reading_three():
    dr = ComparisonTextDataReader(folder="tests/test_data/text/comparison")
    assert dr.folder == "tests/test_data/text/comparison"
    dataset = dr.get_emotion_data("three", Set.TEST, batch_size=4)
    assert isinstance(dataset, tf.data.Dataset)
    batch = 0
    for texts, labels in dataset:
        batch += 1
        if batch < 2:
            assert texts.numpy().shape == (4,)
            assert labels.numpy().shape == (4, 3)
            for text in texts.numpy():
                text = str(text)
                assert len(text) > 5
            for label in labels.numpy():
                assert label.shape == (3,)
                assert np.sum(label) == 1
        elif batch == 2:
            assert texts.numpy().shape == (3,)
            assert labels.numpy().shape == (3, 3)
            for text in texts.numpy():
                text = str(text)
                assert len(text) > 5
            for label in labels.numpy():
                assert label.shape == (3,)
                assert np.sum(label) == 1
    assert batch == 2


def test_labels():
    dr = ComparisonTextDataReader(folder="tests/test_data/text/comparison")
    assert dr.folder == "tests/test_data/text/comparison"
    dataset = dr.get_emotion_data(
        "neutral_ekman", Set.TEST, batch_size=8, parameters={"shuffle": False}
    )
    dataset_labels = np.empty((0,))
    dataset_data = np.empty((0,))
    dataset_raw_labels = np.empty((0, 7))
    for data, labels in dataset:
        dataset_data = np.concatenate([dataset_data, data.numpy()], axis=0)
        labels = labels.numpy()
        dataset_raw_labels = np.concatenate(
            [dataset_raw_labels, labels], axis=0
        )
        labels = np.argmax(labels, axis=1)
        assert labels.shape == (7,)
        dataset_labels = np.concatenate([dataset_labels, labels], axis=0)
    true_labels = dr.get_labels(Set.TEST)
    assert true_labels.shape == (7,)
    assert dataset_labels.shape == (7,)
    assert np.array_equal(true_labels, dataset_labels)
    d_data, d_labels = ComparisonTextDataReader.convert_to_numpy(dataset)
    assert np.array_equal(d_data, dataset_data)
    assert np.array_equal(d_labels, dataset_raw_labels)
