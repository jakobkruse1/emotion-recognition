""" This file tests the evaluator class. """
import pytest

from src.evaluation.evaluator import Evaluator


def test_evaluator_read():
    evaluator = Evaluator()
    evaluator.read_results("tests/test_data/evaluation/results.json")
    assert (
        evaluator.result_paths[0] == "tests/test_data/evaluation/results.json"
    )
    assert len(evaluator.result_paths) == 1
    assert len(evaluator.result_data) == 1
    result = evaluator.result_data[0]
    assert result["modality"] == "text"
    assert result["model"] == "bert"  # Random testing stuff
    assert result["model_name"] == "bert_en_uncased_L-2_H-128_A-2"
    assert result["init_parameters"] is None
    assert isinstance(result["train_parameters"], dict)

    testdata_folder = "tests/test_data/text"
    acc = evaluator.get_scores("accuracy", data_folder=testdata_folder)
    pca = evaluator.get_scores(
        "per_class_accuracy", data_folder=testdata_folder
    )
    pre = evaluator.get_scores("avg_precision", data_folder=testdata_folder)
    rec = evaluator.get_scores("avg_recall", data_folder=testdata_folder)
    with pytest.raises(ValueError):
        _ = evaluator.get_scores("wrong")

    for score in acc, pca, pre, rec:
        for single_score in score:
            assert 0 < single_score <= 1

    evaluator2 = Evaluator()
    evaluator2.read_results("tests/test_data/evaluation/res*.json")
    assert evaluator.get_parameters() == evaluator2.get_parameters()

    evaluator3 = Evaluator()
    evaluator3.read_results(["tests/test_data/evaluation/results.json"])
    assert evaluator.get_parameters() == evaluator3.get_parameters()


def test_evaluator_read_cv_results():
    evaluator = Evaluator()
    evaluator.read_results("tests/test_data/evaluation/cv_results.json")
    assert (
        evaluator.result_paths[0]
        == "tests/test_data/evaluation/cv_results.json"
    )
    assert len(evaluator.result_paths) == 1
    assert len(evaluator.result_data) == 1
    result = evaluator.result_data[0]
    assert result["modality"] == "plant"
    assert result["model"] == "plant_lstm"
    assert result["init_parameters"] is None
    assert isinstance(result["train_parameters"], dict)

    testdata_folder = "tests/test_data/plant"
    acc = evaluator.get_scores("accuracy", data_folder=testdata_folder)
    pca = evaluator.get_scores(
        "per_class_accuracy", data_folder=testdata_folder
    )
    pre = evaluator.get_scores("avg_precision", data_folder=testdata_folder)
    rec = evaluator.get_scores("avg_recall", data_folder=testdata_folder)

    parameters = evaluator.get_parameters()
    for parameter_dict in parameters:
        assert "predictions" not in parameter_dict.keys()
        assert "train_parameters" in parameter_dict.keys()

    for score in acc, pca, pre, rec:
        for single_score in score:
            assert 0 < single_score <= 1
