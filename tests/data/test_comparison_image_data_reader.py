"""Test the comparison image data reader"""

import os

import numpy as np
import pytest
import tensorflow as tf

from src.data.comparison_image_data_reader import (
    ComparisonImageDataReader,
    Set,
)


def test_initialization():
    dr = ComparisonImageDataReader()
    assert dr.name == "comparison_image"
    assert dr.folder == os.path.join("data", "comparison_dataset", "image")


def test_reading():
    dr = ComparisonImageDataReader(
        folder=os.path.join("tests", "test_data", "image", "test")
    )
    assert dr.folder == os.path.join("tests", "test_data", "image", "test")
    dataset = dr.get_emotion_data("neutral_ekman", Set.TEST, batch_size=10)
    assert isinstance(dataset, tf.data.Dataset)
    batch = 0
    for images, labels in dataset:
        batch += 1
        assert images.numpy().shape == (7, 48, 48, 1)
        assert labels.numpy().shape == (7, 7)
        assert np.array_equal(
            labels.numpy()[[0, 6, 1, 3, 2, 5, 4], :], np.eye(7)
        )
    assert batch == 1

    with pytest.raises(AssertionError):
        _ = dr.get_seven_emotion_data(Set.TRAIN)

    with pytest.raises(AssertionError):
        _ = dr.get_seven_emotion_data(Set.VAL)

    with pytest.raises(ValueError):
        _ = dr.get_emotion_data("wrong")


def test_reading_three():
    dr = ComparisonImageDataReader(
        folder=os.path.join("tests", "test_data", "image", "test")
    )
    assert dr.folder == os.path.join("tests", "test_data", "image", "test")
    dataset = dr.get_emotion_data(
        "three", Set.TEST, batch_size=2, parameters={"shuffle": False}
    )
    seven_dataset = dr.get_emotion_data(
        "neutral_ekman", Set.TEST, batch_size=2, parameters={"shuffle": False}
    ).as_numpy_iterator()
    assert isinstance(dataset, tf.data.Dataset)
    batch = 0
    conversion_dict = {0: 2, 1: 0, 2: 2, 3: 0, 4: 2, 5: 2, 6: 1}
    for images, labels in dataset:
        seven_images, seven_labels = next(seven_dataset)
        assert np.array_equal(seven_images, images.numpy())
        batch += 1
        if batch <= 3:
            assert images.numpy().shape == (2, 48, 48, 1)
            assert labels.numpy().shape == (2, 3)
            for index, label in enumerate(labels.numpy()):
                assert (
                    np.argmax(label)
                    == conversion_dict[int(np.argmax(seven_labels[index, :]))]
                )
                assert label.shape == (3,)
                assert np.sum(label) == 1
        elif batch == 4:
            assert images.numpy().shape == (1, 48, 48, 1)
            assert labels.numpy().shape == (1, 3)
            for index, label in enumerate(labels.numpy()):
                assert (
                    np.argmax(label)
                    == conversion_dict[int(np.argmax(seven_labels[index, :]))]
                )
                assert label.shape == (3,)
                assert np.sum(label) == 1
    assert batch == 4


def test_labels():
    dr = ComparisonImageDataReader(
        folder=os.path.join("tests", "test_data", "image", "test")
    )
    assert dr.folder == os.path.join("tests", "test_data", "image", "test")
    dataset = dr.get_emotion_data(
        "neutral_ekman", Set.TEST, batch_size=5, parameters={"shuffle": False}
    )
    dataset_labels = np.empty((0,))
    dataset_data = np.empty((0, 48, 48, 1))
    dataset_raw_labels = np.empty((0, 7))
    for data, labels in dataset:
        dataset_data = np.concatenate([dataset_data, data.numpy()], axis=0)
        labels = labels.numpy()
        dataset_raw_labels = np.concatenate(
            [dataset_raw_labels, labels], axis=0
        )
        labels = np.argmax(labels, axis=1)
        assert labels.shape == (5,) or labels.shape == (2,)
        dataset_labels = np.concatenate([dataset_labels, labels], axis=0)
    true_labels = dr.get_labels(Set.TEST)
    assert true_labels.shape == (7,)
    assert dataset_labels.shape == (7,)
    assert np.array_equal(true_labels, dataset_labels)
    d_data, d_labels = ComparisonImageDataReader.convert_to_numpy(dataset)
    assert np.array_equal(d_data, dataset_data)
    assert np.array_equal(d_labels, dataset_raw_labels)


def test_conversion_function():
    labels = [0, 1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1, 0]
    one_hot_labels = np.eye(7)[labels]
    assert one_hot_labels.shape == (13, 7)
    data, converted = ComparisonImageDataReader.map_emotions(
        "testing", one_hot_labels
    )
    converted_labels = [2, 0, 2, 0, 2, 2, 1, 2, 2, 0, 2, 0, 2]
    assert np.array_equal(np.eye(3)[converted_labels], converted)
    assert data == "testing"
