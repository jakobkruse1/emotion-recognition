"""Experiment running functionality for grid searching parameters
with cross validation ."""

import json
import os

import numpy as np

from src.classification.classifier_factory import ClassifierFactory
from src.data.data_reader import Set
from src.experiment.experiment import Experiment, ExperimentRunner


class CrossValidationExperimentRunner(ExperimentRunner):
    """
    Experiment runner class to run multiple experiments easily
    while also using cross validation
    """

    def __init__(self, experiment_name: str, cv_splits: int = 5, **kwargs):
        """
        Constructor for the ExperimentRunner class
        :param experiment_name: Name of the experiment for log files and result
        :param cv_splits: How many splits to do in cross validation
        :param kwargs: Additional keyword arguments
            Not currently used
        """
        super().__init__(experiment_name, **kwargs)
        self.cv_splits = cv_splits

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

        labels = ClassifierFactory.get(
            experiment.modality, experiment.model, experiment.train_parameters
        ).data_reader.get_labels(Set.ALL)

        # If already exists
        file_path = f"{index:03d}_results.json"
        if os.path.exists(os.path.join(self.folder, file_path)):
            print("Skipping experiment as results already exist!")
            with open(os.path.join(self.folder, file_path), "r") as json_file:
                data = json.load(json_file)
                return np.sum(data["predictions"] == labels) / labels.shape[0]

        predictions = np.empty((0,))

        for cv_split in range(self.cv_splits):
            classifier = ClassifierFactory.get(
                experiment.modality,
                experiment.model,
                experiment.init_parameters,
            )
            train_parameters = experiment.train_parameters.copy()
            train_parameters["cv_portions"] = self.cv_splits
            train_parameters["cv_index"] = cv_split
            classifier.train(train_parameters)
            eval_parameters = train_parameters.copy()
            eval_parameters["which_set"] = Set.TEST
            test_predictions = classifier.classify(eval_parameters)
            predictions = np.concatenate(
                [test_predictions, predictions], axis=0
            )
        parameters = experiment.get_parameter_dict()
        parameters["predictions"] = predictions.tolist()
        with open(os.path.join(self.folder, file_path), "w") as json_file:
            json.dump(parameters, json_file)
        return np.sum(predictions == labels) / labels.shape[0]
