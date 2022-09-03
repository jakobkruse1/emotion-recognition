"""File that contains the evaluator class implementing basic evaluation
functionality and data reading"""

import copy
import glob
import json
from typing import Any, Dict, Iterable, List, Union, Optional

import numpy as np

from src.data.data_factory import DataFactory
from src.data.data_reader import Set


class Evaluator:
    """
    Class that contains basic evaluation functionality
    """

    def __init__(self) -> None:
        """
        Initialization method
        """
        self.result_paths = None
        self.result_data = []
        self.data_readers = {}
        self.precomputed_labels = {}

    def read_results(self, paths: Union[str, Iterable[str]]) -> None:
        """
        Main data reading function that reads all results from files.

        :param paths: Paths argument that can contain different things:
            - A string that contains the path to a single results file
            - A string that is used for glob.glob (must contain at least one *)
            - An iterable (e.g. list) of strings of single files
        """
        if isinstance(paths, str):
            if "*" in paths:
                self.result_paths = list(glob.glob(paths))
            else:
                self.result_paths = [paths]
        else:
            self.result_paths = paths
        for result_path in self.result_paths:
            self._read_result_file(result_path)

    def _read_result_file(self, path: str) -> None:
        """
        Function that reads the results of one file into the class instance.

        :param path: Path to a single results file.
        """
        with open(path, "r") as json_file:
            data = json.load(json_file)
        self.result_data.append(data)

    def get_parameters(self) -> List[Dict]:
        """
        Function that returns the parameters of all experiments in a list.

        :return: List of all experiment configurations without predictions.
        """
        parameters = []
        for experiment in self.result_data:
            data = copy.deepcopy(experiment)
            if "train_predictions" in data.keys():
                del data["train_predictions"]
                del data["val_predictions"]
                del data["test_predictions"]
            else:
                del data["predictions"]
            parameters.append(data)
        return parameters

    def get_labels(
        self, modality: str, predictions_key: str, parameters: Dict[str, Any],
            data_folder: Optional[str] = None
    ) -> np.ndarray:
        """
        Function that returns labels without recomputing them if used before.

        :param modality: The modality of the data.
        :param predictions_key: "test_predictions" or "predicitons"
        :param parameters: Additional parameters required for label generation.
        :param data_folder: Data folder for the data reader.
        :return: Label numpy array.
        """
        critical_parameters = ["label_mode", "window", "hop"]
        which_set = (
            Set.TEST if predictions_key == "test_predictions" else Set.ALL
        )
        if modality not in self.data_readers.keys():
            self.data_readers[modality] = DataFactory.get_data_reader(
                modality, data_folder
            )
            self.precomputed_labels[modality] = []
        contains_critical = np.any(
            [param in parameters.keys() for param in critical_parameters]
        )
        if not contains_critical:
            for label_dict in self.precomputed_labels[modality]:
                if which_set == label_dict["which_set"]:
                    return label_dict["labels"]
            self.precomputed_labels[modality].append(
                {
                    "labels": self.data_readers[modality].get_labels(
                        which_set
                    ),
                    "which_set": which_set,
                }
            )
            self.data_readers[modality].cleanup()
            for label_dict in self.precomputed_labels[modality]:
                if which_set == label_dict["which_set"]:
                    return label_dict["labels"]
        else:
            for label_dict in self.precomputed_labels[modality]:
                if which_set == label_dict["which_set"] and np.all(
                    [
                        label_dict[param] == parameters[param]
                        for param in critical_parameters
                    ]
                ):
                    return label_dict["labels"]
            self.precomputed_labels[modality].append(
                {
                    "labels": self.data_readers[modality].get_labels(
                        which_set, parameters
                    ),
                    "which_set": which_set,
                    **{
                        critical_param: parameters.get(critical_param, None)
                        for critical_param in critical_parameters
                    },
                }
            )
            self.data_readers[modality].cleanup()
            for label_dict in self.precomputed_labels[modality]:
                if which_set == label_dict["which_set"] and np.all(
                    [
                        label_dict[param] == parameters[param]
                        for param in critical_parameters
                    ]
                ):
                    return label_dict["labels"]

    def get_scores(self, score: str, **kwargs) -> List[float]:
        """
        Function that uses all experiments in this evaluator and computes
        a score for them. Returns a list of scores.
        :param score: String that represents the score to compute
        :param kwargs: Additional keyword arguments
        :return: List of computed scores
        """
        scores = []
        predictions_key = (
            "test_predictions"
            if "test_predictions" in self.result_data[0].keys()
            else "predictions"
        )
        data_folder = kwargs.get("data_folder", None)
        for experiment in self.result_data:
            if score == "accuracy":
                scores.append(
                    self._accuracy(
                        self.get_labels(
                            experiment["modality"],
                            predictions_key,
                            experiment["train_parameters"],
                            data_folder
                        ),
                        np.asarray(experiment[predictions_key]),
                    )
                )
            elif score == "avg_recall":
                scores.append(
                    self._avg_recall(
                        self.get_labels(
                            experiment["modality"],
                            predictions_key,
                            experiment["train_parameters"],
                            data_folder
                        ),
                        np.asarray(experiment[predictions_key]),
                    )
                )
            elif score == "avg_precision":
                scores.append(
                    self._avg_precision(
                        self.get_labels(
                            experiment["modality"],
                            predictions_key,
                            experiment["train_parameters"],
                            data_folder
                        ),
                        np.asarray(experiment[predictions_key]),
                    )
                )
            elif score == "per_class_accuracy":
                scores.append(
                    self._per_class_accuracy(
                        self.get_labels(
                            experiment["modality"],
                            predictions_key,
                            experiment["train_parameters"],
                            data_folder
                        ),
                        np.asarray(experiment[predictions_key]),
                    )
                )
        return scores

    @staticmethod
    def _accuracy(true: np.ndarray, pred: np.ndarray) -> float:
        """
        Score function that computes accuracy

        :param true: The true labels
        :param pred: The predicted labels
        :return: The accuracy
        """
        return np.sum(true == pred) / true.shape[0]

    @staticmethod
    def _per_class_accuracy(true: np.ndarray, pred: np.ndarray) -> float:
        """
        Score function that computes accuracy per class and returns an average

        :param true: The true labels
        :param pred: The predicted labels
        :return: The accuracy
        """
        per_class_accs = []
        for class_id in range(7):
            map = true == class_id
            class_predictions = pred[map]
            if class_predictions.size == 0:
                per_class_accs.append(0)
            else:
                per_class_accs.append(
                    np.sum(class_predictions == class_id)
                    / class_predictions.shape[0]
                )
        return np.mean(per_class_accs)

    @staticmethod
    def _avg_recall(true: np.ndarray, pred: np.ndarray) -> float:
        """
        Score function that computes average recall over all classes

        :param true: The true labels
        :param pred: The predicted labels
        :return: The average class recall
        """
        recall = 0.0
        for class_id in range(7):
            recall += (
                np.sum(true[true == class_id] == pred[true == class_id])
                / true[true == class_id].shape[0]
            )
        return recall / 7.0

    @staticmethod
    def _avg_precision(true: np.ndarray, pred: np.ndarray) -> float:
        """
        Score function that computes average precision over all classes

        :param true: The true labels
        :param pred: The predicted labels
        :return: The average class precision
        """
        prec = 0.0
        for class_id in range(7):
            if true[pred == class_id].shape[0]:
                prec += (
                    np.sum(true[pred == class_id] == pred[pred == class_id])
                    / true[pred == class_id].shape[0]
                )
        return prec / 7.0
