"""Test the image data reader"""

import numpy as np
import pytest
import tensorflow as tf

from src.data.image_data_reader import ImageDataReader, Set


def test_initialization():
    dr = ImageDataReader()
    assert dr.name == "image"
    assert dr.folder == "data/train/image"
    for set_type in [Set.TRAIN, Set.VAL, Set.TEST]:
        assert dr.folder_map[set_type] == set_type.name.lower()


def test_reading():
    dr = ImageDataReader(folder="tests/test_data/image")
    assert dr.folder == "tests/test_data/image"
    dataset = dr.get_emotion_data(
        "neutral_ekman",
        Set.TRAIN,
        batch_size=10,
        parameters={"shuffle": False},
    )
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

    with pytest.raises(ValueError):
        _ = dr.get_emotion_data("wrong")


def test_reading_three():
    dr = ImageDataReader(folder="tests/test_data/image")
    assert dr.folder == "tests/test_data/image"
    dataset = dr.get_emotion_data(
        "three", Set.TRAIN, batch_size=2, parameters={"shuffle": False}
    )
    seven_dataset = dr.get_emotion_data(
        "neutral_ekman", Set.TRAIN, batch_size=2, parameters={"shuffle": False}
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
    dr = ImageDataReader(folder="tests/test_data/image")
    dataset = dr.get_emotion_data(
        "neutral_ekman", Set.TRAIN, batch_size=5, parameters={"shuffle": False}
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
    true_labels = dr.get_labels(Set.TRAIN)
    assert true_labels.shape == (7,)
    assert dataset_labels.shape == (7,)
    assert np.array_equal(true_labels, dataset_labels)
    d_data, d_labels = ImageDataReader.convert_to_numpy(dataset)
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
            Set.TRAIN,
            batch_size=7,
            parameters={"shuffle": True},
        )
        dataset_labels = np.empty((0,))
        for _, labels in dataset:
            labels = labels.numpy()
            labels = np.argmax(labels, axis=1)
            dataset_labels = np.concatenate([dataset_labels, labels], axis=0)
            assert labels.shape == (7,)
        trials += 1
        equal = np.array_equal(true_labels, dataset_labels)
    assert not equal


def test_conversion_function():
    labels = [0, 1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1, 0]
    one_hot_labels = np.eye(7)[labels]
    assert one_hot_labels.shape == (13, 7)
    data, converted = ImageDataReader.map_emotions("testing", one_hot_labels)
    converted_labels = [2, 0, 2, 0, 2, 2, 1, 2, 2, 0, 2, 0, 2]
    assert np.array_equal(np.eye(3)[converted_labels], converted)
    assert data == "testing"


def test_augmentation():
    tf.random.set_seed(42)

    dr = ImageDataReader(folder="tests/test_data/image")
    dataset = dr.get_emotion_data(
        "neutral_ekman",
        Set.TRAIN,
        batch_size=5,
        parameters={"shuffle": False, "augment": False},
    )
    augmented_dataset = dr.get_emotion_data(
        "neutral_ekman",
        Set.TRAIN,
        batch_size=5,
        parameters={"shuffle": False, "augment": True},
    )
    for batch, aug_batch in zip(dataset, augmented_dataset):
        images, labels = batch
        aug_images, aug_labels = aug_batch
        assert np.array_equal(labels.numpy(), aug_labels.numpy())
        assert not np.array_equal(images.numpy(), aug_images.numpy())

    counter = 0
    for batch, manual_batch in zip(augmented_dataset, dataset):
        images, labels = batch
        manual_images, manual_labels = dr._augment(
            manual_batch, (counter, counter + 1)
        )
        assert np.array_equal(labels.numpy(), manual_labels.numpy())
        counter += 2
        assert images.shape == manual_images.shape
    assert counter == 4
