"""This script prints the five best image models from the experiments."""

from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.data.data_factory import DataFactory, Set
from src.evaluation.evaluator import Evaluator


def plot_confusion_matrix(model_data, title="Confusion Matrix"):
    labels = DataFactory.get_data_reader("plant").get_labels(
        Set.ALL, parameters=model_data["train_parameters"]
    )
    predictions = np.asarray(model_data["predictions"])

    data = {
        "true": pd.Categorical(labels, categories=list(range(7))),
        "pred": pd.Categorical(predictions, categories=list(range(7))),
    }
    df = pd.DataFrame(data, columns=["true", "pred"])
    confusion_matrix = pd.crosstab(
        df["true"],
        df["pred"],
        rownames=["True"],
        colnames=["Predicted"],
        dropna=False,
    )
    emotions = ["AN", "SU", "DI", "JO", "FE", "SA", "NE"]
    confusion_matrix.index = emotions
    confusion_matrix.columns = emotions
    sns.heatmap(confusion_matrix, annot=True, fmt="d")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.show()


def filter_experiments(
    parameters: List[Dict[str, Any]], key: str, value: Any, default: Any = None
) -> List[int]:
    filtered_indices = []
    for index, experiment in enumerate(parameters):
        if experiment["train_parameters"].get(key, default) == value:
            filtered_indices.append(index)
    return filtered_indices


if __name__ == "__main__":  # pragma: no cover
    # Best models
    evaluator = Evaluator()
    evaluator.read_results("experiments/results/plant_parameters_2/*.json")
    accuracies = evaluator.get_scores("per_class_accuracy")
    parameters = evaluator.get_parameters()
    all_data = evaluator.result_data

    sorted_ind = np.argsort(-np.asarray(accuracies))
    sorted_acc = np.asarray(accuracies)[sorted_ind]
    sorted_params = np.array([parameters[ind] for ind in sorted_ind])
    sorted_data = np.array([all_data[ind] for ind in sorted_ind])

    # Drop all data with weighted=False, because they are useless

    print("++++++++ Best Expected Labels Models ++++++++")
    expected_indices = filter_experiments(
        sorted_params, "label_mode", "expected"
    )
    expected_acc = sorted_acc[expected_indices]
    expected_params = [sorted_params[ind] for ind in expected_indices]
    plot_confusion_matrix(
        sorted_data[expected_indices[0]], "Best expected labels model"
    )
    for i in range(5):
        print(f"Model {i+1}, Per Class Accuracy {expected_acc[i]}")
        print(f"\tParameters: {expected_params[i]}\n")

    print("++++++++ Best Faceapi Labels Models ++++++++")
    faceapi_indices = filter_experiments(
        sorted_params, "label_mode", "faceapi"
    )
    faceapi_acc = sorted_acc[faceapi_indices]
    faceapi_params = [sorted_params[ind] for ind in faceapi_indices]
    plot_confusion_matrix(
        sorted_data[faceapi_indices[0]], "Best faceapi labels model"
    )
    for i in range(5):
        print(f"Model {i + 1}, Per Class Accuracy {faceapi_acc[i]}")
        print(f"\tParameters: {faceapi_params[i]}\n")
