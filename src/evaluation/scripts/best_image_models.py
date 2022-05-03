"""This script prints the five best image models from the experiments."""
import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.data.data_factory import DataFactory, Set
from src.evaluation.evaluator import Evaluator

if __name__ == "__main__":  # pragma: no cover
    evaluator = Evaluator()
    evaluator.read_results(
        glob.glob("experiments/results/efficientnet_parameters/*.json")
        + glob.glob("experiments/results/vgg16_parameters/*.json")
    )
    accuracies = evaluator.get_scores("accuracy")
    sorted_ind = np.argsort(-np.asarray(accuracies))
    sorted_acc = np.asarray(accuracies)[sorted_ind]
    parameters = evaluator.get_parameters()
    sorted_params = [parameters[ind] for ind in sorted_ind]

    for i in range(5):
        print(f"Model {i+1}, Accuracy {sorted_acc[i]}")
        print(f"\tParameters: {sorted_params[i]}\n")

    best_model_id = sorted_ind[0]
    best_model_data = evaluator.result_data[best_model_id]
    predictions = np.asarray(best_model_data["test_predictions"])
    labels = DataFactory.get_data_reader("image").get_labels(Set.TEST)
    data = {"true": labels, "pred": predictions}
    df = pd.DataFrame(data, columns=["true", "pred"])
    confusion_matrix = pd.crosstab(
        df["true"], df["pred"], rownames=["True"], colnames=["Predicted"]
    )
    emotions = ["AN", "SU", "DI", "JO", "FE", "SA", "NE"]
    confusion_matrix.index = emotions
    confusion_matrix.columns = emotions
    sns.heatmap(confusion_matrix, annot=True, fmt="d")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()
