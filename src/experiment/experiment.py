"""Experiment running functionality for grid searching parameters"""

import copy
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
        :param kwargs: Arguments that define the experiments
            See all possible keys below in the list
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

    def check_parameters(self) -> None:
        """
        This function checks the parameters for the experiment.
        This is useful, because if the parameters are not correct, the code
        will crash at some point unexpectedly. Checking the parameters before
        running multiple experiments makes sure that all experiments finish
        successfully.
        This function either throws an Assertion Error or Value Error in case
        of wrong parameters.
        """
        assert self.modality in ["text", "image", "speech", "plant"]
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
            Not currently used
        """
        self.experiments = []
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
            warnings.warn("The experiment folder already exists!")
        os.makedirs(self.folder, exist_ok=True)

    def add_grid_experiments(self, **kwargs):
        """
        Create the cross product of all the arguments lists and create
        experiments from that like in a grid search
        :param kwargs: Keyword arguments which shall be grid searched
            All possible keys can be seen below
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
            experiment_dict = copy.deepcopy(base_dict)
            for index, element in enumerate(grid_search_keys):
                experiment_dict[element] = experiment[index]
            self.experiments.append(Experiment(**experiment_dict))

    def add_single_experiment(self, **kwargs):
        """
        Add an experiment to the experiments list
        :param kwargs: Arguments for the experiment
        """
        self.experiments.append(Experiment(**kwargs))

    def run_all(self, **kwargs):
        """
        Main run function that runs all experiment in the self.experiments list
        :param kwargs: Additional keyword arguments
        """
        self.accuracy = []
        indices = kwargs.get("indices", list(range(len(self.experiments))))
        for index in indices:
            experiment = self.experiments[index]
            accuracy = self.run_experiment(experiment, index, **kwargs)
            self.accuracy.append(accuracy)
            print(experiment.get_parameter_dict())
            print(f"Experiment {index}, Accuracy {accuracy}")
        print("*****\nFinished all runs successfully\n*****")
        self.best_index = np.argmax(np.array(self.accuracy))

    def run_experiment(
        self, experiment: Experiment, index: int, **kwargs
    ) -> float:
        """
        Run an experiment and save the results in a json file

        :param experiment: The experiment configuration to use
        :param index: The index of the experiment for saving the results
        :param kwargs: Additional kwargs
            data_reader: Overwrite data reader for testing purposes
        """
        print(f"Running experiment {index}")
        print(experiment.get_parameter_dict())
        classifier = ClassifierFactory.get(
            experiment.modality, experiment.model, experiment.init_parameters
        )
        classifier.data_reader = kwargs.get(
            "data_reader", classifier.data_reader
        )
        labels = classifier.data_reader.get_labels(
            Set.TEST, experiment.train_parameters
        )
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
        eval_parameters = copy.deepcopy(parameters.get("train_parameters", {}))
        eval_parameters["which_set"] = Set.TRAIN
        eval_parameters["shuffle"] = False
        parameters["train_predictions"] = classifier.classify(
            eval_parameters
        ).tolist()
        eval_parameters["which_set"] = Set.VAL
        parameters["val_predictions"] = classifier.classify(
            eval_parameters
        ).tolist()
        eval_parameters["which_set"] = Set.TEST
        parameters["test_predictions"] = classifier.classify(
            eval_parameters
        ).tolist()
        with open(os.path.join(self.folder, file_path), "w") as json_file:
            json.dump(parameters, json_file)
        return (
            np.sum(parameters["test_predictions"] == labels) / labels.shape[0]
        )


def make_dictionaries(base_dict: Dict = None, **kwargs) -> List[Dict]:
    """
    Create a list of dictionaries from a dictionary of lists inside kwargs

    :param base_dict: The base parameter dictionary to start from
    :param kwargs: All parameters that shall be put in the dictionaries
    :return: List of configuration dictionaries
    """
    all_configs = []
    base_dict = base_dict or {}
    base_dictionary = copy.deepcopy(base_dict)
    grid_search_keys = []
    grid_search_values = []
    for key, value in kwargs.items():
        if isinstance(value, list):
            grid_search_keys.append(key)
            grid_search_values.append(value)
        else:
            base_dictionary[key] = value
    configurations = list(itertools.product(*grid_search_values))

    for config in configurations:
        config_dict = copy.deepcopy(base_dictionary)
        for index, element in enumerate(grid_search_keys):
            config_dict[element] = config[index]
        all_configs.append(config_dict)
    return all_configs
