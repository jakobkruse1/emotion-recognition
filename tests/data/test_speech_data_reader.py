"""Test the speech data reader"""

import numpy as np
import pytest
import tensorflow as tf

from src.data.speech_data_reader import Set, SpeechDataReader


def test_initialization():
    dr = SpeechDataReader()
    assert dr.name == "speech"
    assert dr.folder == "data/train/speech"
    for set_type in [Set.TRAIN, Set.VAL, Set.TEST]:
        assert dr.folder_map[set_type] == set_type.name.lower()


def test_reading():
    dr = SpeechDataReader(folder="tests/test_data/speech")
    assert dr.folder == "tests/test_data/speech"
    dataset = dr.get_emotion_data(
        "neutral_ekman",
        Set.VAL,
        batch_size=7,
        parameters={"shuffle": False},
    )
    num_crema = 738
    assert isinstance(dataset, tf.data.Dataset)
    batch = 0
    for images, labels in dataset:
        batch += 1
        if batch < (738 + 7) // 7:
            assert images.numpy().shape == (7, 48000)
            assert labels.numpy().shape == (7, 7)
        if batch == 1:
            assert np.array_equal(
                labels.numpy()[[0, 6, 1, 3, 2, 5, 4], :], np.eye(7)
            )
    assert batch == (num_crema + 7 + 6) // 7

    with pytest.raises(ValueError):
        _ = dr.get_emotion_data("wrong")


def test_reading_three():
    dr = SpeechDataReader(folder="tests/test_data/speech")
    assert dr.folder == "tests/test_data/speech"
    dataset = dr.get_emotion_data(
        "three",
        Set.VAL,
        batch_size=1,
        parameters={"shuffle": False, "max_elements": 7},
    )
    seven_dataset = dr.get_emotion_data(
        "neutral_ekman",
        Set.VAL,
        batch_size=1,
        parameters={"shuffle": False, "max_elements": 7},
    ).as_numpy_iterator()
    assert isinstance(dataset, tf.data.Dataset)
    batch = 0
    conversion_dict = {0: 2, 1: 0, 2: 2, 3: 0, 4: 2, 5: 2, 6: 1}
    for images, labels in dataset:
        seven_images, seven_labels = next(seven_dataset)
        assert np.array_equal(seven_images, images.numpy())
        batch += 1
        if batch <= 7:
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


def test_labels():
    dr = SpeechDataReader(folder="tests/test_data/speech")
    dataset = dr.get_emotion_data(
        "neutral_ekman", Set.VAL, batch_size=5, parameters={"shuffle": False}
    )
    num_crema = 738
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
    true_labels = dr.get_labels(Set.VAL)

    assert true_labels.shape == (num_crema + 7,)
    assert dataset_labels.shape == (num_crema + 7,)
    assert np.array_equal(true_labels, dataset_labels)
    d_data, d_labels = SpeechDataReader.convert_to_numpy(dataset)
    assert np.array_equal(d_data, dataset_data)
    assert np.array_equal(d_labels, dataset_raw_labels)

    # Now with shuffle
    trials = 0
    equal = True
    while equal:
        if trials > 3:
            raise RuntimeError("Shuffle not working.")
        dataset = dr.get_emotion_data(
            "neutral_ekman",
            Set.VAL,
            batch_size=7,
            parameters={"shuffle": True},
        )
        dataset_labels = np.empty((0,))
        for _, labels in dataset:
            labels = labels.numpy()
            labels = np.argmax(labels, axis=1)
            dataset_labels = np.concatenate([dataset_labels, labels], axis=0)
        trials += 1
        equal = np.array_equal(true_labels, dataset_labels)
    assert not equal


def test_conversion_function():
    labels = [0, 1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1, 0]
    one_hot_labels = np.eye(7)[labels]
    assert one_hot_labels.shape == (13, 7)
    data, converted = SpeechDataReader.map_emotions("testing", one_hot_labels)
    converted_labels = [2, 0, 2, 0, 2, 2, 1, 2, 2, 0, 2, 0, 2]
    assert np.array_equal(np.eye(3)[converted_labels], converted)
    assert data == "testing"


def test_crema_d_dataset():
    num_train = 5144 + 7
    num_test = 1556 + 7
    num_val = 738 + 7

    dr = SpeechDataReader(folder="tests/test_data/speech")
    train_ds = dr.get_labels(Set.TRAIN)
    assert train_ds.shape[0] == num_train
    val_ds = dr.get_labels(Set.VAL)
    assert val_ds.shape[0] == num_val
    test_ds = dr.get_labels(Set.TEST)
    assert test_ds.shape[0] == num_test


def test_dataset_selection():
    dr = SpeechDataReader(folder="tests/test_data/speech")
    val_ds = dr.get_labels(Set.VAL, parameters={"dataset": "meld"})
    assert val_ds.shape[0] == 7
    val_ds = dr.get_labels(Set.VAL, parameters={"dataset": "crema"})
    assert val_ds.shape[0] == 738


def test_get_waveform():
    dr = SpeechDataReader(folder="tests/test_data/speech")
    audio, label = dr.get_waveform_and_label(
        b"tests/test_data/speech/train/angry/03-01-05-01-01-01-02.wav"
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
    dr = SpeechDataReader(folder="tests/test_data/speech")
    audio_raw = (np.random.rand(42000) - 0.5) * 2 * 32768
    audio, label = dr.process_crema(audio_raw, 1)
    assert audio.shape == (48000,)
    assert label.shape == (7,)
    assert label[3] == 1
    for i in list(range(0, 3)) + list(range(4, 7)):
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

    x, y = SpeechDataReader.set_tensor_shapes(audio_tensor, label_tensor)

    assert x.shape.rank == 2
    assert y.shape.rank == 2
