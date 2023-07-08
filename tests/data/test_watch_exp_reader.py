""" Testing the watch experiment data reader class. """
import os

import numpy as np
import pytest
import tensorflow as tf

from src.data.watch_exp_reader import Set, WatchExperimentDataReader


@pytest.mark.filterwarnings("ignore:Happimeter data:UserWarning")
def test_init():
    reader = WatchExperimentDataReader(
        folder=os.path.join("tests", "test_data", "watch"),
        default_label_mode="expected",
    )
    assert reader.name == "watch"
    assert reader.default_label_mode == "expected"
    assert reader.raw_data is None
    assert reader.raw_labels is None
    assert len(reader.get_complete_data_indices()) == 54


@pytest.mark.filterwarnings("ignore:Happimeter data:UserWarning")
def test_get_raw_data():
    reader = WatchExperimentDataReader(
        folder=os.path.join("tests", "test_data", "watch"),
        default_label_mode="expected",
    )
    reader.get_raw_data({})
    assert reader.raw_data.shape == (97, 20, 5)
    assert reader.raw_labels.shape == (97,)
    expected_labels = reader.raw_labels

    reader = WatchExperimentDataReader(
        folder=os.path.join("tests", "test_data", "watch"),
        default_label_mode="faceapi",
    )
    reader.get_raw_data({})
    assert reader.raw_data.shape == (97, 20, 5)
    assert reader.raw_labels.shape == (97,)
    faceapi_labels = reader.raw_labels

    both_labels = expected_labels[expected_labels == faceapi_labels]

    reader = WatchExperimentDataReader(
        folder=os.path.join("tests", "test_data", "watch"),
        default_label_mode="both",
    )
    reader.get_raw_data({})
    assert reader.raw_data.shape == (both_labels.shape[0], 20, 5)
    assert reader.raw_labels.shape == (both_labels.shape[0],)
    assert np.array_equal(both_labels, reader.raw_labels)


@pytest.mark.filterwarnings("ignore:Happimeter data:UserWarning")
def test_normalization():
    reader = WatchExperimentDataReader(
        folder=os.path.join("tests", "test_data", "watch"),
        default_label_mode="expected",
    )
    reader.get_raw_data({})

    assert np.mean(reader.raw_data[:, :, 0], axis=(0, 1)) == pytest.approx(
        0, abs=0.2
    )
    assert np.var(reader.raw_data[:, :, 0], axis=(0, 1)) == pytest.approx(
        1, abs=0.5
    )
    for column in range(1, 4):
        assert -2 < np.mean(reader.raw_data[:, :, column], axis=(0, 1)) < 2
    assert np.mean(reader.raw_data[:, :, 4], axis=(0, 1)) == pytest.approx(
        1, rel=0.2
    )


@pytest.mark.filterwarnings("ignore:Happimeter data:UserWarning")
@pytest.mark.parametrize("window", [10, 20, 30])
def test_input_shape(window):
    true_value = (window, 5)
    reader = WatchExperimentDataReader(
        folder=os.path.join("tests", "test_data", "watch"),
        default_label_mode="expected",
    )
    assert (
        reader.get_input_shape(
            {
                "window": window,
            }
        )
        == true_value
    )


@pytest.mark.filterwarnings("ignore:Happimeter data:UserWarning")
def test_get_data():
    reader = WatchExperimentDataReader(
        folder=os.path.join("tests", "test_data", "watch"),
        default_label_mode="expected",
    )
    data = reader.get_seven_emotion_data(Set.TRAIN, 8, {"shuffle": True})
    assert isinstance(data, tf.data.Dataset)
    for batch, labels in data:
        assert batch.shape[1] == 20
        assert batch.shape[2] == 5
        assert batch.shape[0] == labels.shape[0]
        assert np.all(labels < 7)
        assert np.all(labels >= 0)


@pytest.mark.filterwarnings("ignore:Happimeter data:UserWarning")
def test_get_cv_indices():
    reader = WatchExperimentDataReader(
        folder=os.path.join("tests", "test_data", "watch"),
        default_label_mode="expected",
    )
    reader.get_raw_data({})
    counts = [13, 8, 6, 8, 35, 20, 7]
    order = [6, 3, 2, 0, 1, 5, 4]
    all_cv_indices = reader.get_cross_validation_indices(Set.ALL, {})
    assert len(all_cv_indices) == sum(counts)
    emotion_indices = {}
    start = 0
    for emotion in order:
        emotion_indices[emotion] = list(range(start, start + counts[emotion]))
        start += counts[emotion]
    expected_indices = []
    for split in range(4, -1, -1):
        for emotion in order:
            start = int(counts[emotion] * split / 5)
            end = int(counts[emotion] * (split + 1) / 5)
            indices = list(range(start, end))
            for index in indices:
                expected_indices += [emotion_indices[emotion][index]]
    assert expected_indices == all_cv_indices

    for index in range(5):
        train_ids = reader.get_cross_validation_indices(
            Set.TRAIN, {"cv_index": index}
        )
        val_ids = reader.get_cross_validation_indices(
            Set.VAL, {"cv_index": index}
        )
        test_ids = reader.get_cross_validation_indices(
            Set.TEST, {"cv_index": index}
        )
        train_set = set(train_ids)
        val_set = set(val_ids)
        test_set = set(test_ids)
        assert len(train_set.intersection(val_set)) == 0
        assert len(train_set.intersection(test_set)) == 0
        assert len(test_set.intersection(val_set)) == 0

    expected_labels = reader.raw_labels[expected_indices]
    assert np.array_equal(reader.get_labels(Set.ALL, {}), expected_labels)


@pytest.mark.filterwarnings("ignore:Happimeter data:UserWarning")
def test_reading_three():
    dr = WatchExperimentDataReader(
        folder=os.path.join("tests", "test_data", "watch")
    )
    assert dr.folder == os.path.join("tests", "test_data", "watch")
    dataset = dr.get_emotion_data(
        "three",
        Set.VAL,
        batch_size=1,
        parameters={"shuffle": False},
    )
    seven_dataset = dr.get_emotion_data(
        "neutral_ekman",
        Set.VAL,
        batch_size=1,
        parameters={"shuffle": False},
    ).as_numpy_iterator()
    assert isinstance(dataset, tf.data.Dataset)
    batch = 0
    conversion_dict = {0: 2, 1: 0, 2: 2, 3: 0, 4: 2, 5: 2, 6: 1}
    for three_batch, labels in dataset:
        seven_batch, seven_labels = next(seven_dataset)
        assert np.array_equal(seven_batch, three_batch.numpy())
        batch += 1
        if batch <= 7:
            assert three_batch.numpy().shape == (1, 20, 5)
            assert labels.numpy().shape == (1, 3)
            for index, label in enumerate(labels.numpy()):
                assert (
                    np.argmax(label)
                    == conversion_dict[int(np.argmax(seven_labels[index, :]))]
                )
                assert label.shape == (3,)
                assert np.sum(label) == 1


@pytest.mark.filterwarnings("ignore:Happimeter data:UserWarning")
def test_raw_generator():
    dr = WatchExperimentDataReader(
        folder=os.path.join("tests", "test_data", "watch")
    )
    dr.get_raw_data({})
    generator = dr.get_data_generator(Set.TEST, {})
    for data, labels in generator():
        assert data.shape == (
            20,
            5,
        )
        assert labels.shape == (7,)
