""" Implement training loops for all the classifiers for testing them. """

import copy
import os
import sys
from typing import Any, Dict, Optional

import numpy as np

from src.data.data_reader import Set
from src.utils.metrics import accuracy, per_class_accuracy


def training_loop(
    classifier,
    parameters: Dict[str, Any],
    save_path: Optional[str] = None,
) -> None:  # pragma: no cover
    """
    Implements an example training loop for a normal classifier.
    This trains the classifier, saves the model, then loads the classifier
    and runs inference on the test set. It then prints the accuracy.

    :param classifier: The classifier instance to train here.
    :param parameters: The parameters used for training the classifier.
    :param save_path: The save path to save the classifier at.
    """
    if not os.path.exists(save_path) or "train" in sys.argv:
        classifier.train(parameters)
        classifier.save()

    classifier.load(parameters)
    emotions = classifier.classify(parameters)
    labels = classifier.data_reader.get_labels(Set.TEST, parameters)
    print(f"Labels Shape: {labels.shape}")
    print(f"Emotions Shape: {emotions.shape}")
    print(f"Accuracy: {accuracy(labels, emotions)}")
    print(f"Accuracy: {per_class_accuracy(labels, emotions)}")


def cv_training_loop(
    classifier,
    parameters: Dict[str, Any],
    save_path: Optional[str] = None,
    cv_splits: int = 5,
) -> None:  # pragma: no cover
    """
    Implements an example training loop for a classifier that is running
    in cross validation mode. This means that the dataset is split into
    'cv_splits' subsets, one of them is used for validation, another one for
    testing and the rest for training. This function trains 'cv_splits'
    separate models in order to get one model for every element in the data.
    This trains the classifiers, saves the models, then loads the classifiers
    and runs inference on the test sets. It then prints the accuracy and
    average accuracy of the different classifiers.

    :param cv_splits: How many splits to use for the dataset.
    :param classifier: The classifier instance to train here.
    :param parameters: The parameters used for training the classifier.
    :param save_path: The save path to save the classifier at.
    """
    # Training here
    accuracies = []
    per_class_accuracies = []
    for i in range(cv_splits):
        split_path = f"{save_path}_{i}"
        if not os.path.exists(split_path) or "train" in sys.argv:
            cv_params = copy.deepcopy(parameters)
            cv_params["cv_index"] = i
            cv_params["cv_splits"] = cv_splits
            classifier.train(cv_params)
            classifier.save({"save_path": split_path})
            pred = classifier.classify(cv_params)
            labels = classifier.data_reader.get_labels(Set.TEST, cv_params)
            accuracies.append(accuracy(labels, pred))
            per_class_accuracies.append(per_class_accuracy(labels, pred))
        print(f"Training Acc: {np.mean(accuracies)} | {accuracies}")
        print(
            f"Training Class Acc: {np.mean(per_class_accuracies)} | "
            f"{per_class_accuracies}"
        )

    # Inference here.
    accuracies = []
    per_class_accuracies = []
    for i in range(cv_splits):
        split_path = f"{save_path}_{i}"
        cv_params = copy.deepcopy(parameters)
        cv_params["cv_index"] = i
        cv_params["cv_splits"] = cv_splits
        cv_params["save_path"] = split_path
        parameters.update(cv_params)
        classifier.load(cv_params)
        emotions = classifier.classify(cv_params)
        labels = classifier.data_reader.get_labels(Set.TEST, cv_params)
        print(f"Labels Shape: {labels.shape}")
        print(f"Emotions Shape: {emotions.shape}")
        accuracies.append(accuracy(labels, emotions))
        per_class_accuracies.append(per_class_accuracy(labels, emotions))
        print(
            f"Split {i} Acc: {accuracies[i]}, "
            f"Per class Acc: {per_class_accuracies[i]}"
        )
    print(f"Total Acc: {np.mean(accuracies)} | {accuracies}")
    print(
        f"Total Per Class Acc: {np.mean(per_class_accuracies)} | "
        f"{per_class_accuracies}"
    )


def reader_main(
    reader, parameters: Optional[Dict[str, Any]]
) -> None:  # pragma: no cover
    """
    Main method that can be used to see if a data reader works as expected.
    This creates a dataset and prints the shapes and counts of the classes
    on the test set.

    :param reader: The data reader instance to evaluate.
    :param parameters: Parameters for the data reader.
    """
    dataset = reader.get_seven_emotion_data(Set.TEST, 64, parameters or {})
    first = True
    all_labels = np.empty((0,))
    for data, labels in dataset:
        if first:
            print(f"Data Shape: {data.shape}, Labels Shape: {labels.shape}")
            first = False
        all_labels = np.concatenate(
            [all_labels, np.argmax(labels, axis=1)], axis=0
        )
    print(f"{np.unique(all_labels, return_counts=True)=}")
