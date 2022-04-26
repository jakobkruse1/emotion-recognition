"""File that contains the evaluator class implementing basic evaluation
functionality and data reading"""

import glob
import json
from typing import Dict, Iterable, List, Union

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
        """
        parameters = []
        for experiment in self.result_data:
            data = experiment.copy()
            del data["train_predictions"]
            del data["val_predictions"]
            del data["test_predictions"]
            parameters.append(data)
        return parameters

    def get_scores(self, score: str, **kwargs) -> List[float]:
        """
        Function that uses all experiments in this evaluator and computes
        a score for them. Returns a list of scores.
        :param score: String that represents the score to compute
        :param kwargs: Additional keyword arguments
        :return: List of computed scores
        """
        scores = []
        data_readers = {}
        for experiment in self.result_data:
            if experiment["modality"] not in data_readers.keys():
                data_readers[
                    experiment["modality"]
                ] = DataFactory.get_data_reader(
                    experiment["modality"], **kwargs
                )
            if score == "accuracy":
                scores.append(
                    self._accuracy(
                        data_readers[experiment["modality"]].get_labels(
                            Set.TEST
                        ),
                        np.asarray(experiment["test_predictions"]),
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
