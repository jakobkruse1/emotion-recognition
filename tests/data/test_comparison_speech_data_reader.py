"""Test the comparison speech data reader"""

import os

import numpy as np
import pytest
import tensorflow as tf

from src.data.comparison_speech_data_reader import (
    ComparisonSpeechDataReader,
    Set,
)


def test_initialization():
    dr = ComparisonSpeechDataReader()
    assert dr.name == "comparison_speech"
    assert dr.folder == os.path.join("data", "comparison_dataset", "audio")


@pytest.mark.filterwarnings("ignore:Truncating audio:UserWarning")
def test_reading():
    dr = ComparisonSpeechDataReader(
        folder=os.path.join("tests", "test_data", "speech", "train")
    )
    assert dr.folder == os.path.join("tests", "test_data", "speech", "train")
    dataset = dr.get_emotion_data(
        "neutral_ekman",
        Set.TEST,
        batch_size=10,
        parameters={"shuffle": False},
    )
    assert isinstance(dataset, tf.data.Dataset)
    batch = 0
    for images, labels in dataset:
        batch += 1
        assert images.numpy().shape == (7, 48000)
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


@pytest.mark.filterwarnings("ignore:Truncating audio:UserWarning")
def test_reading_three():
    dr = ComparisonSpeechDataReader(
        folder=os.path.join("tests", "test_data", "speech", "train")
    )
    assert dr.folder == os.path.join("tests", "test_data", "speech", "train")
    dataset = dr.get_emotion_data(
        "three",
        Set.TEST,
        batch_size=1,
    )
    seven_dataset = dr.get_emotion_data(
        "neutral_ekman",
        Set.TEST,
        batch_size=1,
    ).as_numpy_iterator()
    assert isinstance(dataset, tf.data.Dataset)
    batch = 0
    conversion_dict = {0: 2, 1: 0, 2: 2, 3: 0, 4: 2, 5: 2, 6: 1}
    for images, labels in dataset:
        seven_images, seven_labels = next(seven_dataset)
        assert np.array_equal(seven_images, images.numpy())
        batch += 1
        assert images.numpy().shape == (1, 48000)
        assert labels.numpy().shape == (1, 3)
        for index, label in enumerate(labels.numpy()):
            assert (
                np.argmax(label)
                == conversion_dict[int(np.argmax(seven_labels[index, :]))]
            )
            assert label.shape == (3,)
            assert np.sum(label) == 1
    assert batch == 7


@pytest.mark.filterwarnings("ignore:Truncating audio:UserWarning")
def test_labels():
    dr = ComparisonSpeechDataReader(
        folder=os.path.join("tests", "test_data", "speech", "train")
    )
    dataset = dr.get_emotion_data(
        "neutral_ekman", Set.TEST, batch_size=5, parameters={"shuffle": False}
    )
    dataset_labels = np.empty((0,))
    dataset_data = np.empty((0, 48000))
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
    d_data, d_labels = ComparisonSpeechDataReader.convert_to_numpy(dataset)
    assert np.array_equal(d_data, dataset_data)
    assert np.array_equal(d_labels, dataset_raw_labels)


def test_conversion_function():
    labels = [0, 1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1, 0]
    one_hot_labels = np.eye(7)[labels]
    assert one_hot_labels.shape == (13, 7)
    data, converted = ComparisonSpeechDataReader.map_emotions(
        "testing", one_hot_labels
    )
    converted_labels = [2, 0, 2, 0, 2, 2, 1, 2, 2, 0, 2, 0, 2]
    assert np.array_equal(np.eye(3)[converted_labels], converted)
    assert data == "testing"


def test_get_waveform():
    dr = ComparisonSpeechDataReader(
        folder=os.path.join("tests", "test_data", "speech", "train")
    )
    with pytest.warns(UserWarning):
        audio, label = dr.get_waveform_and_label(
            os.path.join(
                "tests",
                "test_data",
                "speech",
                "train",
                "angry",
                "03-01-05-01-01-01-02.wav",
            ).encode()
        )
    audio = audio.numpy()
    label = label.numpy()
    assert audio.shape == (48000,)
    assert label.shape == (7,)
    assert label[0] == 1
    for i in range(1, 7):
        assert label[i] == 0
    assert np.max(audio) <= 1
    assert np.min(audio) >= -1


def test_tensor_shapes():
    audio = np.random.rand(1, 48000)
    label = np.zeros((1, 7))
    label[0, 3] = 1
    audio_tensor = tf.convert_to_tensor(audio)
    label_tensor = tf.convert_to_tensor(label)
    audio_tensor.set_shape(tf.TensorShape(None))
    label_tensor.set_shape(tf.TensorShape(None))

    x, y = ComparisonSpeechDataReader.set_tensor_shapes(
        audio_tensor, label_tensor
    )

    assert x.shape.rank == 2
    assert y.shape.rank == 2
