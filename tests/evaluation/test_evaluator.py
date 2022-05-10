""" This file tests the evaluator class. """

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
    assert result["modality"] == "image"
    assert result["model"] == "bert"  # Random testing stuff
    assert result["model_name"] == "bert_en_uncased_L-2_H-128_A-2"
    assert result["init_parameters"] is None
    assert isinstance(result["train_parameters"], dict)

    evaluator2 = Evaluator()
    evaluator2.read_results("tests/test_data/evaluation/*.json")
    assert evaluator.get_parameters() == evaluator2.get_parameters()

    evaluator3 = Evaluator()
    evaluator3.read_results(["tests/test_data/evaluation/results.json"])
    assert evaluator.get_parameters() == evaluator3.get_parameters()


def test_evaluator_score_accuracy():
    evaluator = Evaluator()
    evaluator.read_results("tests/test_data/evaluation/results.json")
    accuracy = evaluator.get_scores(
        "accuracy", data_folder="tests/test_data/image"
    )

    assert len(accuracy) == 1
    # True Labels [0, 6, 1, 3, 2, 5, 4]
    assert accuracy[0] == 1 / 7.0


def test_evaluator_score_avg_recall():
    evaluator = Evaluator()
    evaluator.read_results("tests/test_data/evaluation/results.json")
    avg_recall = evaluator.get_scores(
        "avg_recall", data_folder="tests/test_data/image"
    )

    assert len(avg_recall) == 1
    assert avg_recall[0] == 1 / 7.0


def test_evaluator_score_avg_precision():
    evaluator = Evaluator()
    evaluator.read_results("tests/test_data/evaluation/results.json")
    avg_precision = evaluator.get_scores(
        "avg_precision", data_folder="tests/test_data/image"
    )

    assert len(avg_precision) == 1
    assert avg_precision[0] == 1 / 7.0
