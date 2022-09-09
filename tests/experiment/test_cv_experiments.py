"""Tests for the cross validation experiments and CVExperimentRunner classes"""
import os
import shutil

import pytest

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
    assert os.path.exists("experiments/results/test_name")
    with pytest.warns(UserWarning):
        _ = CrossValidationExperimentRunner("test_name", 7)
    shutil.rmtree("experiments/results/test_name", ignore_errors=True)


@pytest.mark.parametrize("cv_splits", [2, 3, 4])
def test_run_cv(cv_splits):
    cv_runner = CrossValidationExperimentRunner("test_name", cv_splits)
    assert cv_runner.cv_splits == cv_splits
    experiment = Experiment(
        modality="text", model="nrclex", train_parameters={}
    )
    cv_runner.experiments.append(experiment)

    shutil.rmtree("experiments/results/test_name", ignore_errors=True)
