""" Testing the plant experiment data reader class. """
import numpy as np
import pytest
import tensorflow as tf

from src.data.plant_exp_reader import PlantExperimentDataReader, Set


def test_init():
    reader = PlantExperimentDataReader(
        folder="tests/test_data/plant", default_label_mode="expected"
    )
    assert reader.name == "plant_exp"
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
    data = reader.get_seven_emotion_data(Set.TRAIN, 8, {"shuffle": False})
    assert isinstance(data, tf.data.Dataset)
