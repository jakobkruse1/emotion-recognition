"""Test the text data reader"""

import numpy as np
import pytest
import tensorflow as tf

from src.data.text_data_reader import Set, TextDataReader


def test_initialization():
    dr = TextDataReader()
    assert dr.name == "text"
    assert dr.folder == "data/train/text"
    for set_type in [Set.TRAIN, Set.VAL, Set.TEST]:
        assert dr.file_map[set_type]


def test_reading():
    dr = TextDataReader(folder="tests/test_data/text")
    assert dr.folder == "tests/test_data/text"
    dataset = dr.get_emotion_data("neutral_ekman", Set.TRAIN, batch_size=5)
    assert isinstance(dataset, tf.data.Dataset)
    batch = 0
    for texts, labels in dataset:
        batch += 1
        assert texts.numpy().shape == (5, 1)
        assert labels.numpy().shape == (5, 7)
        for text in texts.numpy():
            text = str(text)
            assert len(text) > 5
        for label in labels.numpy():
            assert label.shape == (7,)
            assert np.sum(label) == 1
    assert batch == 6

    with pytest.raises(ValueError):
        _ = dr.get_emotion_data("wrong")


def test_reading_three():
    dr = TextDataReader(folder="tests/test_data/text")
    assert dr.folder == "tests/test_data/text"
    dataset = dr.get_emotion_data("three", Set.TRAIN, batch_size=4)
    assert isinstance(dataset, tf.data.Dataset)
    batch = 0
    for texts, labels in dataset:
        batch += 1
        if batch <= 7:
            assert texts.numpy().shape == (4, 1)
            assert labels.numpy().shape == (4, 3)
            for text in texts.numpy():
                text = str(text)
                assert len(text) > 5
            for label in labels.numpy():
                assert label.shape == (3,)
                assert np.sum(label) == 1
        elif batch == 8:
            assert texts.numpy().shape == (2, 1)
            assert labels.numpy().shape == (2, 3)
            for text in texts.numpy():
                text = str(text)
                assert len(text) > 5
            for label in labels.numpy():
                assert label.shape == (3,)
                assert np.sum(label) == 1
    assert batch == 8


def test_labels():
    dr = TextDataReader(folder="tests/test_data/text")
    dataset = dr.get_emotion_data(
        "neutral_ekman", Set.TRAIN, batch_size=5, shuffle=False
    )
    dataset_labels = np.empty((0,))
    dataset_data = np.empty((0, 1))
    dataset_raw_labels = np.empty((0, 7))
    for data, labels in dataset:
        dataset_data = np.concatenate([dataset_data, data.numpy()], axis=0)
        labels = labels.numpy()
        dataset_raw_labels = np.concatenate(
            [dataset_raw_labels, labels], axis=0
        )
        labels = np.argmax(labels, axis=1)
        assert labels.shape == (5,)
        dataset_labels = np.concatenate([dataset_labels, labels], axis=0)
    true_labels = dr.get_labels(Set.TRAIN)
    assert true_labels.shape == (30,)
    assert dataset_labels.shape == (30,)
    assert np.array_equal(true_labels, dataset_labels)
    d_data, d_labels = TextDataReader.convert_to_numpy(dataset)
    assert np.array_equal(d_data, dataset_data)
    assert np.array_equal(d_labels, dataset_raw_labels)

    # Now with shuffle
    dataset = dr.get_emotion_data(
        "neutral_ekman", Set.TRAIN, batch_size=5, shuffle=True
    )
    dataset_labels = np.empty((0,))
    for _, labels in dataset:
        labels = labels.numpy()
        labels = np.argmax(labels, axis=1)
        assert labels.shape == (5,)
        dataset_labels = np.concatenate([dataset_labels, labels], axis=0)
    assert not np.array_equal(true_labels, dataset_labels)
