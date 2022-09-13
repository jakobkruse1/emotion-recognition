""" Implement interesting metrics to use them later. """

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
)


def accuracy(labels: np.ndarray, prediction: np.ndarray, **kwargs) -> float:
    """
    Compute the accuracy of predictions.

    :param labels: Labels array.
    :param prediction: Predicitions array from classifier.
    :param kwargs: Additional kwargs.
    :return: Accuracy
    """
    return accuracy_score(labels, prediction)


def per_class_accuracy(
    labels: np.ndarray, prediction: np.ndarray, **kwargs
) -> float:
    """
    Compute the average per class accuracy of predictions.

    :param labels: Labels array.
    :param prediction: Predicitions array from classifier.
    :param kwargs: Additional kwargs.
    :return: Per class accuracy average
    """
    matrix = confusion_matrix(labels, prediction)
    per_class_accs = matrix.diagonal() / matrix.sum(axis=1)
    return np.nanmean(per_class_accs)


def precision(labels: np.ndarray, prediction: np.ndarray, **kwargs) -> float:
    """
    Compute the precision of predictions.

    :param labels: Labels array.
    :param prediction: Predicitions array from classifier.
    :param kwargs: Additional kwargs.
    :return: Precision
    """
    return precision_score(
        labels, prediction, average="macro", zero_division=0
    )


def recall(labels: np.ndarray, prediction: np.ndarray, **kwargs) -> float:
    """
    Compute the recall of predictions.

    :param labels: Labels array.
    :param prediction: Predicitions array from classifier.
    :param kwargs: Additional kwargs.
    :return: Recall
    """
    return recall_score(labels, prediction, average="macro")
