""" Test the balanced watch reader for correctness. """
import numpy as np
import pytest

from src.data.balanced_watch_exp_reader import (
    BalancedWatchExperimentDataReader,
)
from src.data.watch_exp_reader import Set, WatchExperimentDataReader


def test_init():
    reader = BalancedWatchExperimentDataReader(folder="tests/test_data/watch")
    assert reader.folder == "tests/test_data/watch"
    assert isinstance(reader.unbalanced_reader, WatchExperimentDataReader)


@pytest.mark.filterwarnings("ignore:Happimeter data:UserWarning")
def test_unbalanced_data():
    reader = BalancedWatchExperimentDataReader(folder="tests/test_data/watch")
    unbalanced_reader = WatchExperimentDataReader(
        folder="tests/test_data/watch"
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
    reader = BalancedWatchExperimentDataReader(folder="tests/test_data/watch")
    parameters = {"balanced": True, "shuffle": False}
    with pytest.raises(NotImplementedError):
        reader.get_three_emotion_data(Set.VAL, 16, parameters)


@pytest.mark.filterwarnings("ignore:Happimeter data:UserWarning")
def test_balanced_data():
    counts = {index: 0 for index in range(7)}
    reader = BalancedWatchExperimentDataReader(folder="tests/test_data/watch")
    parameters = {"balanced": True, "shuffle": True}
    dataset = reader.get_seven_emotion_data(Set.TRAIN, 16, parameters)
    total_count = 55
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
