"""This script prints the five best image models from the experiments."""
import glob

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from src.evaluation.evaluator import Evaluator

if __name__ == "__main__":  # pragma: no cover
    evaluator = Evaluator()
    evaluator.read_results(
        glob.glob("experiments/results/image_augment_parameters/*.json")
    )
    accuracies = evaluator.get_scores("accuracy")
    recalls = evaluator.get_scores("avg_recall")
    parameters = evaluator.get_parameters()
    markers = [
        "o" if params["train_parameters"]["weighted"] else "x"
        for params in parameters
    ]
    colors = [
        "b" if params["train_parameters"]["augment"] else "r"
        for params in parameters
    ]
    for i in range(len(accuracies)):
        plt.scatter(
            accuracies[i], recalls[i], color=colors[i], marker=markers[i]
        )
    handles = [
        mpatches.Patch(color="b", label="Augment True"),
        mpatches.Patch(color="r", label="Augment False"),
        Line2D(
            [0],
            [0],
            color="w",
            markerfacecolor="b",
            marker="o",
            label="Weighted True",
            markersize=15,
        ),
        Line2D(
            [0],
            [0],
            color="w",
            markeredgecolor="b",
            marker="x",
            label="Weighted False",
            markersize=15,
        ),
    ]
    plt.xlabel("Accuracy")
    plt.ylabel("Recall")
    plt.legend(handles=handles)
    plt.grid()
    plt.show()
