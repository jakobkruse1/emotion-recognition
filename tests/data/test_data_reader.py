"""Test the data reader interface and set definitions"""

import numpy as np
import pytest
import tensorflow as tf

from src.data.data_reader import DataReader, Set


def test_sets():
    assert 0 == Set.TRAIN.value
    assert 1 == Set.VAL.value
    assert 2 == Set.TEST.value


def test_data_reader():
    with pytest.raises(TypeError):
        # Cannot instantiate interface class
        _ = DataReader("test", "folder")

    seven_emotions = np.array([0, 1, 2, 3, 4, 5, 6, 6, 5, 4, 3, 2, 1, 0])
    converted = DataReader.convert_to_three_emotions(seven_emotions)
    assert np.array_equal(
        converted, np.array([2, 0, 2, 0, 2, 2, 1, 1, 2, 2, 0, 2, 0, 2])
    )


def test_numpy_conversion():
    data = np.random.randn(80, 15, 3)
    labels = np.random.randn(80, 4)
    dataset = tf.data.Dataset.from_tensor_slices((data, labels))
    dataset = dataset.batch(9)
    b_data, b_labels = DataReader.convert_to_numpy(dataset)
    assert np.array_equal(b_data, data)
    assert np.array_equal(b_labels, labels)


def test_onehot_conversion():
    labels = [0, 1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1, 0]
    one_hot_labels = np.eye(7)[labels]
    assert one_hot_labels.shape == (13, 7)
    converted = DataReader.convert_to_three_emotions_onehot(one_hot_labels)
    converted_labels = [2, 0, 2, 0, 2, 2, 1, 2, 2, 0, 2, 0, 2]
    assert np.array_equal(np.eye(3)[converted_labels], converted)

    with pytest.raises(AssertionError):
        _ = DataReader.convert_to_three_emotions_onehot(np.zeros((6, 6)))
