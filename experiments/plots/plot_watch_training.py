""" Plot statistics and confusion matrix from the best watch model. """
import copy
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.classification.watch import WatchRandomForestClassifier
from src.data.data_reader import Set


def plot_confusion_matrix(predictions, labels):
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
        normalize="index",
    )
    emotions = ["AN", "SU", "DI", "JO", "FE", "SA", "NE"]
    confusion_matrix.index = emotions
    confusion_matrix.columns = emotions

    plt.figure(figsize=(4, 3))
    sns.heatmap(confusion_matrix, annot=True, fmt=".2f", cmap="Blues")
    plt.xlabel("Predicted emotion")
    plt.ylabel("True emotion")
    plt.tight_layout()
    plt.savefig("plots/watch_confusion.pdf")
    plt.show()


def main():
    os.makedirs("plots", exist_ok=True)
    matplotlib.rcParams.update({"font.size": 11})
    plt.rc("text", usetex=True)
    plt.rc("font", family="serif")

    classifier = WatchRandomForestClassifier()
    parameters = {
        "label_mode": "both",
        "batch_size": 64,
        "window": 20,
        "hop": 2,
        "balanced": True,
        "max_depth": 30,
        "n_estimators": 10,
        "min_samples_split": 4,
    }
    save_path = "models/watch/random_forest"
    labels = np.empty((0,))
    predictions = np.empty((0,))
    for i in range(5):
        split_path = f"{save_path}_{i}"
        cv_params = copy.deepcopy(parameters)
        cv_params["cv_index"] = i
        cv_params["cv_splits"] = 5
        cv_params["save_path"] = split_path
        classifier.load(cv_params)
        split_predictions = classifier.classify(cv_params)
        split_labels = classifier.data_reader.get_labels(Set.TEST, cv_params)
        labels = np.concatenate([labels, split_labels], axis=0)
        predictions = np.concatenate([predictions, split_predictions], axis=0)

    plot_confusion_matrix(predictions, labels)


if __name__ == "__main__":
    main()
