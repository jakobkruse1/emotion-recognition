"""Tests for the cross validation experiments and CVExperimentRunner classes"""
import json
import os
import shutil

import pytest

from src.classification.plant.dense_classifier import PlantDenseClassifier
from src.data.plant_exp_reader import PlantExperimentDataReader, Set
from src.experiment.cv_experiment import (
    CrossValidationExperimentRunner,
    Experiment,
)


def test_init():
    cv_runner = CrossValidationExperimentRunner("test_name", 8)
    assert cv_runner.base_experiment_name == "test_name"
    assert cv_runner.cv_splits == 8
    assert cv_runner.accuracy is None
    assert cv_runner.best_index is None
    assert len(cv_runner.experiments) == 0
    assert os.path.exists(os.path.join("experiments", "results", "test_name"))
    with pytest.warns(UserWarning):
        _ = CrossValidationExperimentRunner("test_name", 7)
    shutil.rmtree(
        os.path.join("experiments", "results", "test_name"), ignore_errors=True
    )


@pytest.mark.parametrize("cv_splits", [2, 3, 4])
def test_run_cv(cv_splits):
    cv_runner = CrossValidationExperimentRunner("test_name", cv_splits)
    assert cv_runner.cv_splits == cv_splits
    experiment = Experiment(
        modality="text", model="nrclex", train_parameters={}
    )
    cv_runner.experiments.append(experiment)

    shutil.rmtree(
        os.path.join("experiments", "results", "test_name"), ignore_errors=True
    )


def test_experiment_runner_run_all(monkeypatch):
    shutil.rmtree(
        os.path.join("experiments", "results", "test_name"), ignore_errors=True
    )

    runner = CrossValidationExperimentRunner("test_name")

    assert len(runner.experiments) == 0
    assert runner.base_experiment_name == "test_name"
    assert os.path.join("experiments", "results", "test_name") in runner.folder

    runner.add_grid_experiments(
        modality="plant",
        model="plant_dense",
        train_parameters=[
            {
                "epochs": 1,
                "dense_units": 64,
                "downsampling_factor": 1000,
                "dense_layers": 1,
                "dropout": 0.0,
                "window": 10,
                "a": 0,
            },
            {
                "epochs": 1,
                "dense_units": 64,
                "downsampling_factor": 1000,
                "dense_layers": 1,
                "dropout": 0.0,
                "window": 10,
                "a": 1,
            },
        ],
    )
    assert len(runner.experiments) == 2

    data_reader = PlantExperimentDataReader(
        folder=os.path.join("tests", "test_data", "plant")
    )

    def train_overwrite(self, parameters, **kwargs):
        self.initialize_model(parameters)

    monkeypatch.setattr(PlantDenseClassifier, "train", train_overwrite)
    runner.run_all(data_reader=data_reader)
    assert runner.best_index is not None
    assert isinstance(runner.accuracy, list)
    assert len(runner.accuracy) == 2

    save_file = os.path.join(runner.folder, "000_results.json")
    assert os.path.exists(save_file)
    with open(save_file, "r") as file:
        data = json.load(file)
        assert data["train_parameters"]["a"] == 0
        assert len(data["predictions"]) == len(data_reader.get_labels(Set.ALL))
    with pytest.warns(UserWarning):
        new_runner = CrossValidationExperimentRunner("test_name")
    new_runner.add_grid_experiments(
        modality="plant",
        model="plant_dense",
        train_parameters=[
            {
                "epochs": 1,
                "dense_units": 64,
                "downsampling_factor": 1000,
                "dense_layers": 1,
                "dropout": 0.0,
                "window": 10,
                "a": 0,
            },
            {
                "epochs": 1,
                "dense_units": 64,
                "downsampling_factor": 1000,
                "dense_layers": 1,
                "dropout": 0.0,
                "window": 10,
                "a": 1,
            },
        ],
    )
    new_runner.run_all(data_reader=data_reader)
    for new_acc, acc in zip(new_runner.accuracy, runner.accuracy):
        assert new_acc == acc
    shutil.rmtree(
        os.path.join("experiments", "results", "test_name"), ignore_errors=True
    )
