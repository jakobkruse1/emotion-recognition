""" Testing the plant experiment data reader class. """
import numpy as np
import pytest
import tensorflow as tf

from src.data.plant_exp_reader import PlantExperimentDataReader, Set


def test_init():
    reader = PlantExperimentDataReader(
        folder="tests/test_data/plant", default_label_mode="expected"
    )
    assert reader.name == "plant"
    assert reader.default_label_mode == "expected"
    assert len(reader.files) == 1
    assert reader.raw_data is None
    assert reader.raw_labels is None
    assert reader.sample_rate == 10000


def test_get_raw_data():
    reader = PlantExperimentDataReader(
        folder="tests/test_data/plant", default_label_mode="expected"
    )
    reader.get_raw_data({})
    assert reader.raw_data.shape == (121, 100000)
    assert reader.raw_labels.shape == (121,)
    expected_labels = reader.raw_labels

    reader = PlantExperimentDataReader(
        folder="tests/test_data/plant", default_label_mode="faceapi"
    )
    reader.get_raw_data({})
    assert reader.raw_data.shape == (121, 100000)
    assert reader.raw_labels.shape == (121,)
    faceapi_labels = reader.raw_labels

    both_labels = expected_labels[expected_labels == faceapi_labels]

    reader = PlantExperimentDataReader(
        folder="tests/test_data/plant", default_label_mode="both"
    )
    reader.get_raw_data({})
    assert reader.raw_data.shape == (both_labels.shape[0], 100000)
    assert reader.raw_labels.shape == (both_labels.shape[0],)
    assert np.array_equal(both_labels, reader.raw_labels)


def test_preprocess():
    reader = PlantExperimentDataReader(
        folder="tests/test_data/plant", default_label_mode="expected"
    )
    data = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 6, 2, 2, 2, 3, 6, -1])
    processed = reader.preprocess_sample(data, {"downsampling_factor": 5})
    assert np.array_equal(np.array([0, 2, 3, -1]), processed)


@pytest.mark.parametrize("preprocess", [True, False])
@pytest.mark.parametrize("window", [10, 20, 30])
def test_input_shape(preprocess, window):
    true_value = window * 10000 if not preprocess else window * 10000 / 5
    reader = PlantExperimentDataReader(
        folder="tests/test_data/plant", default_label_mode="expected"
    )
    assert (
        reader.get_input_shape(
            {
                "downsampling_factor": 5,
                "preprocess": preprocess,
                "window": window,
            }
        )[0]
        == true_value
    )


def test_cv_indices():
    reader = PlantExperimentDataReader(
        folder="tests/test_data/plant", default_label_mode="expected"
    )
    parameters = {"window": 10, "hop": 5}
    reader.get_raw_data(parameters)


def test_get_data():
    reader = PlantExperimentDataReader(
        folder="tests/test_data/plant", default_label_mode="expected"
    )
    data = reader.get_seven_emotion_data(Set.TRAIN, 8, {"shuffle": True})
    assert isinstance(data, tf.data.Dataset)
    for batch, labels in data:
        assert batch.shape[1] == 200
        assert batch.shape[0] == labels.shape[0]
        assert np.all(labels < 7)
        assert np.all(labels >= 0)
    assert reader.raw_data is not None
    assert reader.raw_labels is not None
    reader.cleanup()
    assert not hasattr(reader, "raw_data")
    assert not hasattr(reader, "raw_labels")


def test_get_cv_indices():
    reader = PlantExperimentDataReader(
        folder="tests/test_data/plant", default_label_mode="expected"
    )
    reader.get_raw_data({})
    counts = [16, 12, 10, 11, 39, 24, 9]  # Counts for the class labels
    order = [6, 3, 2, 0, 1, 5, 4]
    emotion_indices = {}
    start = 0
    for emotion in order:
        emotion_indices[emotion] = list(range(start, start + counts[emotion]))
        start += counts[emotion]
    all_cv_indices = reader.get_cross_validation_indices(Set.ALL, {})
    all_cumulative = []
    for cv_index in range(5):
        print(cv_index)
        # Test indices
        expected = []
        for emotion in order:
            indices = emotion_indices[emotion]
            expected.extend(
                indices[
                    int((0.8 - cv_index / 5) * len(indices)) : int(
                        (1.0 - cv_index / 5) * len(indices) + 0.000001
                    )
                ]
            )
        print(emotion_indices)
        print(expected)
        assert (
            reader.get_cross_validation_indices(
                Set.TEST, {"cv_index": cv_index}
            )
            == expected
        )
        assert (
            reader.get_cross_validation_indices(
                Set.VAL, {"cv_index": (cv_index + 4) % 5}
            )
            == expected
        )
        all_cumulative = all_cumulative + expected
    assert all_cumulative == all_cv_indices

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


def test_reading_three():
    dr = PlantExperimentDataReader(folder="tests/test_data/plant")
    assert dr.folder == "tests/test_data/plant"
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
    for plant_batch, labels in dataset:
        seven_plant, seven_labels = next(seven_dataset)
        assert np.array_equal(seven_plant, plant_batch.numpy())
        batch += 1
        if batch <= 7:
            assert plant_batch.numpy().shape == (1, 200)
            assert labels.numpy().shape == (1, 3)
            for index, label in enumerate(labels.numpy()):
                assert (
                    np.argmax(label)
                    == conversion_dict[int(np.argmax(seven_labels[index, :]))]
                )
                assert label.shape == (3,)
                assert np.sum(label) == 1


def test_raw_generator():
    dr = PlantExperimentDataReader(folder="tests/test_data/plant")
    dr.get_raw_data({})
    generator = dr.get_data_generator(Set.TEST, {})
    for data, labels in generator():
        assert data.shape == (200,)
        assert labels.shape == (7,)
