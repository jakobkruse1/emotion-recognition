""" Plot statistics and confusion matrix from the best text model. """

import os

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.classification.text import DistilBertClassifier
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

    plt.figure(figsize=(5, 4))
    sns.heatmap(confusion_matrix, annot=True, fmt=".2f", cmap="Blues")
    plt.xlabel("Predicted emotion")
    plt.ylabel("True emotion")
    plt.tight_layout()
    plt.savefig("plots/distilbert_confusion.pdf")
    plt.show()


def main():
    os.makedirs("plots", exist_ok=True)
    matplotlib.rcParams.update({"font.size": 12})
    plt.rc("text", usetex=True)
    plt.rc("font", family="serif")

    classifier = DistilBertClassifier()
    parameters = {"init_lr": 1e-05, "dropout_rate": 0.2, "dense_layer": 1024}
    save_path = "models/text/distilbert"
    parameters.update({"save_path": save_path})
    classifier.load(parameters)

    prediction = classifier.classify(parameters)
    labels = classifier.data_reader.get_labels(Set.TEST, parameters)

    plot_confusion_matrix(prediction, labels)


if __name__ == "__main__":
    main()
