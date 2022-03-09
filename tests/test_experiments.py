"""Tests for the experiments and ExperimentRunner classes"""
import json
import os
import shutil

import pytest

from src.data.data_reader import Set
from src.data.text_data_reader import TextDataReader
from src.experiment.experiment import (
    Experiment,
    ExperimentRunner,
    make_dictionaries,
)


def test_experiment():
    experiment = Experiment(
        init_parameters={"a": 2},
        train_parameters={"b": 3},
        model_name="bert_123",
        model="bert",
        modality="text",
    )
    assert experiment.init_parameters == {"a": 2}
    assert experiment.train_parameters == {"b": 3}
    assert experiment.model_name == "bert_123"
    assert experiment.model == "bert"
    assert experiment.modality == "text"

    with pytest.raises(AssertionError):
        _ = Experiment(stupid_param="test")

    with pytest.raises(ValueError):
        _ = Experiment(modality="text", model="not_exist")

    with pytest.raises(AssertionError):
        _ = Experiment(train_parameters=3)
        _ = Experiment(init_parameters=3)


def test_experiment_runner_configs(monkeypatch):
    shutil.rmtree("experiments/results/test_name", ignore_errors=True)

    _ = ExperimentRunner("test_name")
    # Same runner again - should ask user for input
    monkeypatch.setattr("builtins.input", lambda _: "n")
    with pytest.raises(SystemExit) as error:
        _ = ExperimentRunner("test_name")
    assert error.type == SystemExit
    assert error.value.code == 0

    monkeypatch.setattr("builtins.input", lambda _: "y")
    runner = ExperimentRunner("test_name")

    assert len(runner.experiments) == 0
    assert runner.experiment_index == 0
    assert runner.base_experiment_name == "test_name"
    assert "experiments/results/test_name" in runner.folder

    runner.add_grid_experiments(
        modality="text",
        model=["bert", "distilbert", "nrclex"],
        train_parameters=[{"a": 2}, {"a": 3}],
    )
    assert len(runner.experiments) == 6
    assert runner.experiments[0].modality == "text"
    assert runner.experiments[0].model == "bert"
    assert runner.experiments[0].train_parameters["a"] == 2

    assert runner.experiments[4].modality == "text"
    assert runner.experiments[4].model == "nrclex"
    assert runner.experiments[4].train_parameters["a"] == 2

    runner.add_single_experiment(
        modality="text", model="bert", train_parameters={"a": 5}
    )
    assert runner.experiments[-1].train_parameters["a"] == 5
    assert len(runner.experiments) == 7

    parameters = runner.experiments[-1].get_parameter_dict()
    assert isinstance(parameters, dict)
    assert parameters["modality"] == "text"
    assert parameters["model"] == "bert"
    assert parameters["train_parameters"]["a"] == 5
    assert parameters["init_parameters"] is None
    assert parameters["model_name"] is None

    shutil.rmtree("experiments/results/test_name", ignore_errors=True)


def test_experiment_runner_run_all():
    shutil.rmtree("experiments/results/test_name", ignore_errors=True)

    runner = ExperimentRunner("test_name")

    assert len(runner.experiments) == 0
    assert runner.experiment_index == 0
    assert runner.base_experiment_name == "test_name"
    assert "experiments/results/test_name" in runner.folder

    runner.add_grid_experiments(
        modality="text", model="nrclex", train_parameters=[{"a": 0}, {"a": 1}]
    )
    assert len(runner.experiments) == 2

    # TODO: Reduce runtime by not using full dataset, only use test data
    runner.run_all()
    assert runner.best_index is not None
    assert isinstance(runner.accuracy, list)
    assert len(runner.accuracy) == 2

    data_reader = TextDataReader()

    for index in range(2):
        save_file = os.path.join(runner.folder, f"{index:03d}_results.json")
        assert os.path.exists(save_file)
        with open(save_file, "r") as file:
            data = json.load(file)
            assert data["train_parameters"]["a"] == index % 2
            assert len(data["train_predictions"]) == len(
                data_reader.get_labels(Set.TRAIN)
            )
            assert len(data["val_predictions"]) == len(
                data_reader.get_labels(Set.VAL)
            )
            assert len(data["test_predictions"]) == len(
                data_reader.get_labels(Set.TEST)
            )

    shutil.rmtree("experiments/results/test_name", ignore_errors=True)


def test_make_dictionaries():
    configs = make_dictionaries(
        base_value="base", first=[1, 2, 3], base2=42, second=[4, 6, 9]
    )
    first_list = [1, 2, 3]
    second_list = [4, 6, 9]
    assert len(configs) == 9
    for index, config in enumerate(configs):
        assert config["base_value"] == "base"
        assert config["base2"] == 42
        assert config["first"] == first_list[index // 3]
        assert config["second"] == second_list[index % 3]
