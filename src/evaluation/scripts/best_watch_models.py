"""This script prints the five best watch models from the experiments."""
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
        if key not in experiment.keys():
            if experiment["train_parameters"].get(key, default) == value:
                filtered_indices.append(index)
        else:
            if experiment.get(key, default) == value:
                filtered_indices.append(index)
    return filtered_indices


def print_best_models(accs, pcaccs, params, data, indices, name) -> None:
    """
    Visualization function that prints and plots certain results

    :param accs: Accuracies
    :param pcaccs: Per Class Accuracies
    :param params: Parameters
    :param data: All results data
    :param indices: Indices to filter for
    :param name: Name of the config to look at
    """
    if not len(indices):
        return
    print(f"++++++++ Best {name} models ++++++++")
    expected_pcacc = pcaccs[indices]
    expected_acc = accs[indices]
    expected_params = [params[ind] for ind in indices]
    plot_confusion_matrix(data[indices[0]], f"Best {name} model")
    for i in range(5):
        print(
            f"Model {i + 1}, Per Class Accuracy {expected_pcacc[i]}, "
            f"Accuracy {expected_acc[i]}"
        )
        print(f"\tParameters: {expected_params[i]}\n")


if __name__ == "__main__":  # pragma: no cover
    # Best models
    evaluator = Evaluator()
    evaluator.read_results("experiments/results/watch_parameters/*.json")
    per_class_accuracies = evaluator.get_scores("per_class_accuracy")
    accuracies = evaluator.get_scores("accuracy")
    parameters = evaluator.get_parameters()
    all_data = evaluator.result_data

    sorted_ind = np.argsort(-np.asarray(per_class_accuracies))
    sorted_pcacc = np.asarray(per_class_accuracies)[sorted_ind]
    sorted_acc = np.asarray(accuracies)[sorted_ind]
    sorted_params = [parameters[ind] for ind in sorted_ind]
    sorted_data = np.array([all_data[ind] for ind in sorted_ind])

    # Drop all data with weighted=False, because they are useless

    expected_indices = filter_experiments(
        sorted_params, "label_mode", "expected"
    )
    print_best_models(
        sorted_acc,
        sorted_pcacc,
        sorted_params,
        sorted_data,
        expected_indices,
        "expected labels",
    )

    faceapi_indices = filter_experiments(
        sorted_params, "label_mode", "faceapi"
    )
    print_best_models(
        sorted_acc,
        sorted_pcacc,
        sorted_params,
        sorted_data,
        faceapi_indices,
        "faceapi labels",
    )

    both_indices = filter_experiments(sorted_params, "label_mode", "both")
    print_best_models(
        sorted_acc,
        sorted_pcacc,
        sorted_params,
        sorted_data,
        both_indices,
        "both labels",
    )
