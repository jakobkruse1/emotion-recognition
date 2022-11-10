""" Plot statistics and confusion matrix from the best image model. """
import json
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

    plt.figure(figsize=(4, 3))
    sns.heatmap(confusion_matrix, annot=True, fmt=".2f", cmap="Blues")
    plt.xlabel("Predicted emotion")
    plt.ylabel("True emotion")
    plt.tight_layout()
    plt.savefig("plots/vgg16_confusion.pdf")
    plt.show()


def plot_training_progress():
    with open("models/image/vgg16/statistics.json", "r") as json_file:
        train_data = json.load(json_file)
    patience = train_data["train_parameters"].get("patience", None)
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot(train_data["train_loss"], label="Loss", c="blue")
    ax.plot(train_data["val_loss"], label="Val. Loss", c="blue", ls="dotted")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    ax2 = ax.twinx()
    ax2.plot(train_data["train_acc"], label="Accuracy", c="orange")
    ax2.plot(train_data["val_acc"], label="Val. Acc.", c="orange", ls="dotted")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Accuracy")

    # Patience line
    if patience:
        location = len(train_data["train_acc"]) - patience - 1
        ax.axvline(location, c="black", ls="dashed")

    # Legend and plot
    leg = ax.legend(loc="lower left")
    leg.remove()
    ax2.legend(loc="upper left")
    ax2.add_artist(leg)
    ax.xaxis.get_major_locator().set_params(integer=True)
    plt.tight_layout()
    plt.savefig("plots/image_training.pdf")
    plt.show()


def main():
    os.makedirs("plots", exist_ok=True)
    matplotlib.rcParams.update({"font.size": 11})
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

    plot_training_progress()


if __name__ == "__main__":
    main()
