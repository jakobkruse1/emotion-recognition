""" Plot statistics and confusion matrix from the best image model. """

import os

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.classification.image import VGG16Classifier
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
    plt.savefig("plots/vgg16_confusion.pdf")
    plt.show()


def main():
    os.makedirs("plots", exist_ok=True)
    matplotlib.rcParams.update({"font.size": 12})
    plt.rc("text", usetex=True)
    plt.rc("font", family="serif")

    classifier = VGG16Classifier()
    parameters = {
        "batch_size": 64,
        "learning_rate": 0.0001,
        "deep": True,
        "dropout": 0.4,
        "frozen_layers": 0,
        "l1": 0,
        "l2": 1e-05,
        "augment": True,
        "weighted": False,
        "balanced": False,
    }
    save_path = "models/image/vgg16"
    parameters.update({"save_path": save_path})
    classifier.load(parameters)

    prediction = classifier.classify(parameters)
    labels = classifier.data_reader.get_labels(Set.TEST, parameters)

    plot_confusion_matrix(prediction, labels)


if __name__ == "__main__":
    main()
