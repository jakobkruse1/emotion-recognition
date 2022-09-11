""" Test the balanced plant reader for correctness. """
import numpy as np
import pytest

from src.data.balanced_plant_exp_reader import (
    BalancedPlantExperimentDataReader,
)
from src.data.plant_exp_reader import PlantExperimentDataReader, Set


def test_init():
    reader = BalancedPlantExperimentDataReader(folder="tests/test_data/plant")
    assert reader.folder == "tests/test_data/plant"
    assert isinstance(reader.unbalanced_reader, PlantExperimentDataReader)


def test_unbalanced_data():
    reader = BalancedPlantExperimentDataReader(folder="tests/test_data/plant")
    unbalanced_reader = PlantExperimentDataReader(
        folder="tests/test_data/plant"
    )
    parameters = {"balanced": False, "shuffle": False}

    balanced_dataset = reader.get_seven_emotion_data(Set.TRAIN, 16, parameters)
    unbalanced_dataset = unbalanced_reader.get_seven_emotion_data(
        Set.TRAIN, 16, parameters
    ).as_numpy_iterator()

    for bal_data, bal_labels in balanced_dataset:
        unb_data, unb_labels = next(unbalanced_dataset)
        assert np.array_equal(bal_data, unb_data)
        assert np.array_equal(unb_labels, bal_labels)

    balanced_dataset = reader.get_three_emotion_data(Set.TRAIN, 16, parameters)
    unbalanced_dataset = unbalanced_reader.get_three_emotion_data(
        Set.TRAIN, 16, parameters
    ).as_numpy_iterator()

    for bal_data, bal_labels in balanced_dataset:
        unb_data, unb_labels = next(unbalanced_dataset)
        assert np.array_equal(bal_data, unb_data)
        assert np.array_equal(unb_labels, bal_labels)

    balanced_labels = reader.get_labels(Set.TEST)
    unbalanced_labels = unbalanced_reader.get_labels(Set.TEST)
    assert np.array_equal(balanced_labels, unbalanced_labels)


def test_balanced_three():
    reader = BalancedPlantExperimentDataReader(folder="tests/test_data/plant")
    parameters = {"balanced": True, "shuffle": False}
    with pytest.raises(NotImplementedError):
        reader.get_three_emotion_data(Set.VAL, 16, parameters)


def test_balanced_data():
    counts = {index: 0 for index in range(7)}
    reader = BalancedPlantExperimentDataReader(folder="tests/test_data/plant")
    parameters = {"balanced": True, "shuffle": True}
    dataset = reader.get_seven_emotion_data(Set.TRAIN, 16, parameters)
    total_count = 70
    iterations = 100
    for iteration in range(iterations):
        for _, labels in dataset:
            for emotion in range(7):
                counts[emotion] += np.count_nonzero(
                    np.argmax(labels, axis=1) == emotion
                )
    print(counts)
    for emotion in range(7):
        assert counts[emotion] == pytest.approx(
            total_count * iterations / 7, rel=0.2
        )

    assert reader.unbalanced_reader.raw_data is not None
    assert reader.unbalanced_reader.raw_labels is not None
    reader.cleanup()
    assert not hasattr(reader.unbalanced_reader, "raw_data")
    assert not hasattr(reader.unbalanced_reader, "raw_labels")
