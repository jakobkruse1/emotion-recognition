"""Test the speech data reader"""

import os

import numpy as np

from src.data.classwise_speech_data_reader import (
    ClasswiseSpeechDataReader,
    Set,
)


def test_initialization():
    dr = ClasswiseSpeechDataReader()
    assert dr.name == "classwise_speech"
    assert dr.folder == "data/train/speech"
    for set_type in [Set.TRAIN, Set.VAL, Set.TEST]:
        assert dr.folder_map[set_type] == set_type.name.lower()


def test_reading():
    CLASS_NAMES = {
        "angry": 127,
        "surprise": 1,
        "disgust": 127,
        "happy": 127,
        "fear": 127,
        "sad": 127,
        "neutral": 109,
    }
    dr = ClasswiseSpeechDataReader(folder="tests/test_data/speech")
    assert dr.folder == "tests/test_data/speech"
    dataset = dr.get_emotion_data(
        "neutral_ekman",
        Set.VAL,
        parameters={"shuffle": True},
    )
    for data, class_name in dataset:
        print(f"{class_name}, {data.shape}")
        assert isinstance(class_name, str)
        assert class_name in CLASS_NAMES.keys()
        assert isinstance(data, np.ndarray)
        assert data.shape == (CLASS_NAMES[class_name], 48000)


def test_reading_three():
    counts = {"positive": 128, "neutral": 109, "negative": 508}
    dr = ClasswiseSpeechDataReader(folder="tests/test_data/speech")
    assert dr.folder == "tests/test_data/speech"
    dataset = dr.get_emotion_data(
        "three",
        Set.VAL,
        parameters={"shuffle": False},
    )
    for data, class_name in dataset:
        assert isinstance(class_name, str)
        assert class_name in ["positive", "neutral", "negative"]
        assert isinstance(data, np.ndarray)
        assert data.shape == (counts[class_name], 48000)


def test_labels():
    counts = [127, 1, 127, 127, 127, 127, 109]
    dr = ClasswiseSpeechDataReader(folder="tests/test_data/speech")
    labels = dr.get_labels(Set.VAL)
    true_labels = np.empty((0,))
    for index, count in enumerate(counts):
        true_labels = np.concatenate(
            [true_labels, np.ones((count,)) * index], axis=0
        )
    assert np.array_equal(labels, true_labels)


def test_meld_only():
    counts = [1, 1, 1, 1, 1, 1, 1]
    dr = ClasswiseSpeechDataReader(folder="tests/test_data/speech")
    labels = dr.get_labels(Set.VAL, parameters={"dataset": "meld"})
    true_labels = np.empty((0,))
    for index, count in enumerate(counts):
        true_labels = np.concatenate(
            [true_labels, np.ones((count,)) * index], axis=0
        )
    assert np.array_equal(labels, true_labels)


def test_crema_only():
    counts = [126, 0, 126, 126, 126, 126, 108]
    dr = ClasswiseSpeechDataReader(folder="tests/test_data/speech")
    labels = dr.get_labels(Set.VAL, parameters={"dataset": "crema"})
    true_labels = np.empty((0,))
    for index, count in enumerate(counts):
        true_labels = np.concatenate(
            [true_labels, np.ones((count,)) * index], axis=0
        )
    assert np.array_equal(labels, true_labels)


def test_conversion_function():
    labels = [0, 1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1, 0]
    one_hot_labels = np.eye(7)[labels]
    assert one_hot_labels.shape == (13, 7)
    data, converted = ClasswiseSpeechDataReader.map_emotions(
        "testing", one_hot_labels
    )
    converted_labels = [2, 0, 2, 0, 2, 2, 1, 2, 2, 0, 2, 0, 2]
    assert np.array_equal(np.eye(3)[converted_labels], converted)
    assert data == "testing"


def test_get_waveform():
    dr = ClasswiseSpeechDataReader(folder="tests/test_data/speech")
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


def test_process_crema():
    dr = ClasswiseSpeechDataReader(folder="tests/test_data/speech")
    audio_raw = (np.random.rand(42000) - 0.5) * 2 * 32768
    audio, label = dr.process_crema(audio_raw, 1)
    assert audio.shape == (48000,)
    assert label.shape == (7,)
    assert label[3] == 1
    for i in list(range(0, 3)) + list(range(4, 7)):
        assert label[i] == 0
    assert np.max(audio) <= 1
    assert np.min(audio) >= -1
