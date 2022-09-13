""" Test metrics file. """

import numpy as np

from src.utils.metrics import accuracy, per_class_accuracy, precision, recall


def test_accuracy():
    assert (
        accuracy(
            np.array([0, 0, 1, 1, 1, 1, 1]), np.array([0, 0, 0, 1, 1, 1, 1])
        )
        == 6.0 / 7
    )
    assert (
        accuracy(
            np.array([0, 0, 0, 1, 1, 1, 1]), np.array([0, 0, 0, 1, 1, 1, 1])
        )
        == 1
    )
    assert (
        accuracy(
            np.array([0, 0, 0, 1, 1, 1, 1]), np.array([1, 1, 1, 0, 0, 0, 0])
        )
        == 0
    )
    assert (
        accuracy(
            np.array([0, 1, 2, 3, 2, 1, 0]), np.array([0, 0, 0, 1, 1, 1, 1])
        )
        == 2.0 / 7
    )
    assert (
        accuracy(
            np.array([0, 1, 2, 3, 2, 1, 0]), np.array([0, 1, 2, 3, 2, 1, 1])
        )
        == 6.0 / 7
    )
    assert (
        accuracy(
            np.array([0, 1, 2, 3, 2, 1, 0]), np.array([0, 1, 2, 3, 2, 1, 0])
        )
        == 1
    )


def test_per_class_accuracy():
    assert (
        per_class_accuracy(
            np.array([0, 0, 1, 1, 1, 1, 1]), np.array([0, 0, 0, 1, 1, 1, 1])
        )
        == 9.0 / 10
    )
    assert (
        per_class_accuracy(
            np.array([0, 0, 0, 1, 1, 1, 1]), np.array([0, 0, 0, 1, 1, 1, 1])
        )
        == 1
    )
    assert (
        per_class_accuracy(
            np.array([0, 0, 0, 1, 1, 1, 1]), np.array([1, 1, 1, 0, 0, 0, 0])
        )
        == 0
    )
    assert (
        per_class_accuracy(
            np.array([0, 1, 2, 3, 2, 1, 0]), np.array([0, 0, 0, 1, 1, 1, 1])
        )
        == 0.25
    )
    assert (
        per_class_accuracy(
            np.array([0, 1, 2, 3, 2, 1, 0]), np.array([0, 1, 2, 3, 2, 1, 1])
        )
        == 7.0 / 8
    )
    assert (
        per_class_accuracy(
            np.array([0, 1, 2, 3, 2, 1, 0]), np.array([0, 1, 2, 3, 2, 1, 0])
        )
        == 1
    )


def test_recall():
    assert (
        recall(
            np.array([0, 0, 1, 1, 1, 1, 1]), np.array([0, 0, 0, 1, 1, 1, 1])
        )
        == 9.0 / 10.0
    )
    assert (
        recall(
            np.array([0, 0, 0, 1, 1, 1, 1]), np.array([0, 0, 0, 1, 1, 1, 1])
        )
        == 1
    )
    assert (
        recall(
            np.array([0, 0, 0, 1, 1, 1, 1]), np.array([1, 1, 1, 0, 0, 0, 0])
        )
        == 0
    )
    assert (
        recall(
            np.array([0, 1, 2, 3, 2, 1, 0]), np.array([0, 0, 0, 1, 1, 1, 1])
        )
        == 0.25
    )
    assert (
        recall(
            np.array([0, 1, 2, 3, 2, 1, 0]), np.array([0, 1, 2, 3, 2, 1, 1])
        )
        == 7.0 / 8
    )
    assert (
        recall(
            np.array([0, 1, 2, 3, 2, 1, 0]), np.array([0, 1, 2, 3, 2, 1, 0])
        )
        == 1
    )


def test_precision():
    assert (
        abs(
            precision(
                np.array([0, 0, 1, 1, 1, 1, 1]),
                np.array([0, 0, 0, 1, 1, 1, 1]),
            )
            - 5.0 / 6
        )
        < 1e-9
    )
    assert (
        precision(
            np.array([0, 0, 0, 1, 1, 1, 1]), np.array([0, 0, 0, 1, 1, 1, 1])
        )
        == 1
    )
    assert (
        precision(
            np.array([0, 0, 0, 1, 1, 1, 1]), np.array([1, 1, 1, 0, 0, 0, 0])
        )
        == 0
    )
    assert (
        abs(
            precision(
                np.array([0, 1, 2, 3, 2, 1, 0]),
                np.array([0, 0, 0, 1, 1, 1, 1]),
            )
            - 7.0 / 48
        )
        < 1e-9
    )
    assert (
        precision(
            np.array([0, 1, 2, 3, 2, 1, 0]), np.array([0, 1, 2, 3, 2, 1, 1])
        )
        == 11.0 / 12
    )
    assert (
        precision(
            np.array([0, 1, 2, 3, 2, 1, 0]), np.array([0, 1, 2, 3, 2, 1, 0])
        )
        == 1
    )
