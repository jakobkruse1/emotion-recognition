"""This script prints the five best text models from the experiments."""
import glob

import numpy as np

from src.evaluation.evaluator import Evaluator

if __name__ == "__main__":  # pragma: no cover
    evaluator = Evaluator()
    evaluator.read_results(
        glob.glob("experiments/results/bert_models/*.json")
        + glob.glob("experiments/results/distilbert_parameters/*.json")
    )
    accuracies = evaluator.get_scores("accuracy")
    sorted_ind = np.argsort(-np.asarray(accuracies))
    sorted_acc = np.asarray(accuracies)[sorted_ind]
    parameters = evaluator.get_parameters()
    sorted_params = [parameters[ind] for ind in sorted_ind]

    for i in range(5):
        print(f"Model {i+1}, Accuracy {sorted_acc[i]}")
        print(f"\tParameters: {sorted_params[i]}\n")
