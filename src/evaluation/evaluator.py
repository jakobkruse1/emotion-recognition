"""File that contains the evaluator class implementing basic evaluation
functionality and data reading"""

import copy
import glob
import json
from typing import Any, Dict, Iterable, List, Optional, Union

import numpy as np
from alive_progress import alive_bar

from src.data.data_factory import DataFactory
from src.data.data_reader import Set
from src.utils.metrics import accuracy, per_class_accuracy, precision, recall


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
        self,
        modality: str,
        predictions_key: str,
        parameters: Dict[str, Any],
        data_folder: Optional[str] = None,
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
        with alive_bar(
            len(self.result_data), title="Computing scores", force_tty=True
        ) as bar:
            for experiment in self.result_data:
                if score == "accuracy":
                    scores.append(
                        accuracy(
                            self.get_labels(
                                experiment["modality"],
                                predictions_key,
                                experiment["train_parameters"],
                                data_folder,
                            ),
                            np.asarray(experiment[predictions_key]),
                        )
                    )
                elif score == "avg_recall":
                    scores.append(
                        recall(
                            self.get_labels(
                                experiment["modality"],
                                predictions_key,
                                experiment["train_parameters"],
                                data_folder,
                            ),
                            np.asarray(experiment[predictions_key]),
                        )
                    )
                elif score == "avg_precision":
                    scores.append(
                        precision(
                            self.get_labels(
                                experiment["modality"],
                                predictions_key,
                                experiment["train_parameters"],
                                data_folder,
                            ),
                            np.asarray(experiment[predictions_key]),
                        )
                    )
                elif score == "per_class_accuracy":
                    scores.append(
                        per_class_accuracy(
                            self.get_labels(
                                experiment["modality"],
                                predictions_key,
                                experiment["train_parameters"],
                                data_folder,
                            ),
                            np.asarray(experiment[predictions_key]),
                        )
                    )
                else:
                    raise ValueError(f"Score {score} does not exist!")
                bar()
        return scores
