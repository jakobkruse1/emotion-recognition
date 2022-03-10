"""Experiment running functionality for grid searching parameters"""

import itertools
import json
import os
import warnings
from typing import Dict, List

import numpy as np

from src.classification.classifier_factory import ClassifierFactory
from src.data.data_reader import Set


class Experiment:
    """
    Experiment class containing the parameters for one run
    """

    def __init__(self, **kwargs):
        """
        This constructor is called to create an experiment instance.
        parses all kwargs to create a fully defined experiment. All parameters
        which are not in the kwargs are set to default values here.
        :param kwargs: Arguments the define the experiments
        """
        # First, set all parameters to default values
        self.possible_keys = [
            "modality",
            "model",
            "model_name",
            "init_parameters",
            "train_parameters",
        ]
        self.modality = None
        self.model = None
        self.model_name = None
        self.train_parameters = None
        self.init_parameters = None

        # Now, parse the kwargs to overwrite defaults
        for key, value in kwargs.items():
            # Check the kwargs for wrong keys
            assert key in self.possible_keys
            setattr(self, key, value)

        # Check the validity of all parameters to avoid late crashing
        self.check_parameters()

    def get_parameter_dict(self) -> Dict:
        """
        Return a dictionary with all the parameters for this experiment

        :return: Dictionary of parameters
        """
        parameters = {}
        for key in self.possible_keys:
            parameters[key] = getattr(self, key, None)
        return parameters

    def check_parameters(self):
        """
        This function checks the parameters for the experiment.
        This is useful, because if the parameters are not correct, the code
        will crash at some point unexpectedly. Checking the parameters before
        running multiple experiments makes sure that all experiments finish
        successfully.
        This function either throws an Assertion Error or Value Error in case
        of wrong parameters.
        """
        assert self.modality in ["text"]
        _ = ClassifierFactory.get(self.modality, self.model, {})
        assert (
            isinstance(self.train_parameters, dict)
            or self.train_parameters is None
        )
        assert (
            isinstance(self.init_parameters, dict)
            or self.init_parameters is None
        )


class ExperimentRunner:
    """
    Experiment runner class to run multiple experiments easily
    """

    def __init__(self, experiment_name: str, **kwargs):
        """
        Constructor for the ExperimentRunner class
        :param experiment_name: Name of the experiment for log files and result
        :param kwargs: Additional keyword arguments
        """
        self.experiments = []
        self.experiment_index = 0
        self.base_experiment_name = experiment_name
        self.best_index = None
        self.accuracy = None
        base_folder = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        self.folder = os.path.join(
            base_folder, f"experiments/results/{experiment_name}"
        )
        if os.path.exists(self.folder):
            warnings.warn(
                "The experiment folder already exists! "
                "Files might be overwritten!"
            )
            self._check_with_user()
        os.makedirs(self.folder, exist_ok=True)

    @staticmethod
    def _check_with_user():
        """
        Function requiring user input to proceed.
        The user needs to input yes or y on the keyboard otherwise the program
        is terminated.
        """
        yes = ["yes", "y"]

        choice = input("Do you want to continue? [Y/N] : ").lower()
        if choice in yes:
            return
        else:
            exit(0)

    def add_grid_experiments(self, **kwargs):
        """
        Create the cross product of all the arguments lists and create
        experiments from that like in a grid search
        :param kwargs: Keyword arguments which shall be grid searched
        """
        for key, _ in kwargs.items():
            # Check the kwargs for wrong keys
            assert key in [
                "modality",
                "model",
                "model_name",
                "train_parameters",
                "init_parameters",
            ]

        base_dict = {}
        grid_search_keys = []
        grid_search_values = []
        for key, value in kwargs.items():
            if isinstance(value, list):
                grid_search_keys.append(key)
                grid_search_values.append(value)
            else:
                base_dict[key] = value
        experiments = list(itertools.product(*grid_search_values))

        for experiment in experiments:
            experiment_dict = base_dict.copy()
            for index, element in enumerate(grid_search_keys):
                experiment_dict[element] = experiment[index]
            self.experiments.append(Experiment(**experiment_dict))

    def add_single_experiment(self, **kwargs):
        """
        Add an experiment to the experiments list
        :param kwargs: Arguments for the experiment
        """
        self.experiments.append(Experiment(**kwargs))

    def run_all(self):
        """
        Main run function that runs all experiment in the self.experiments list
        """
        self.accuracy = []
        for experiment in self.experiments:
            accuracy = self.run_experiment(experiment, self.experiment_index)
            self.accuracy.append(accuracy)
            self.experiment_index += 1
            print(
                f"{self.experiment_index}/{len(self.experiments)} : "
                f"Accuracy {accuracy}"
            )
        print("*****\nFinished all runs successfully\n*****")
        self.best_index = np.argmax(np.array(self.accuracy))
        print(
            f"Best: Index {self.best_index}, "
            f"Acc {self.accuracy[self.best_index]}, "
            f"Parameters "
            f"{self.experiments[self.best_index].get_parameter_dict()}"
        )

    def run_experiment(self, experiment: Experiment, index: int) -> float:
        """
        Run an experiment and save the results in a json file

        :param experiment: The experiment configuration to use
        :param index: The index of the experiment for saving the results
        """
        classifier = ClassifierFactory.get(
            experiment.modality, experiment.model, experiment.init_parameters
        )
        labels = classifier.data_reader.get_labels(Set.TEST)
        file_path = f"{index:03d}_results.json"
        if os.path.exists(os.path.join(self.folder, file_path)):
            print("Skipping experiment as results already exist!")
            with open(os.path.join(self.folder, file_path), "r") as json_file:
                data = json.load(json_file)
                return (
                    np.sum(data["test_predictions"] == labels)
                    / labels.shape[0]
                )

        classifier.train(experiment.train_parameters)
        parameters = experiment.get_parameter_dict()
        parameters["train_predictions"] = classifier.classify(
            {"set": Set.TRAIN}
        ).tolist()
        parameters["val_predictions"] = classifier.classify(
            {"set": Set.VAL}
        ).tolist()
        parameters["test_predictions"] = classifier.classify(
            {"set": Set.TEST}
        ).tolist()
        with open(os.path.join(self.folder, file_path), "w") as json_file:
            json.dump(parameters, json_file)

        return (
            np.sum(parameters["test_predictions"] == labels) / labels.shape[0]
        )


def make_dictionaries(**kwargs) -> List[Dict]:
    """
    Create a list of dictionaries from a dictionary of lists inside kwargs

    :param kwargs: All parameters that shall be put in the dictionaries
    :return: List of configuration dictionaries
    """
    all_configs = []
    base_dict = {}
    grid_search_keys = []
    grid_search_values = []
    for key, value in kwargs.items():
        if isinstance(value, list):
            grid_search_keys.append(key)
            grid_search_values.append(value)
        else:
            base_dict[key] = value
    configurations = list(itertools.product(*grid_search_values))

    for config in configurations:
        config_dict = base_dict.copy()
        for index, element in enumerate(grid_search_keys):
            config_dict[element] = config[index]
        all_configs.append(config_dict)
    return all_configs
