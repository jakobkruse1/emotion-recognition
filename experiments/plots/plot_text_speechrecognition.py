""" Visualize how the text classifiers performs when using
speech recognition. """
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import confusion_matrix

from src.classification.classifier_factory import ClassifierFactory
from src.data.data_factory import DataFactory
from src.data.data_reader import Set


def score_per_class(
    labels: np.ndarray, prediction: np.ndarray
) -> dict[str, float]:
    matrix = confusion_matrix(labels, prediction)
    avg_recalls = matrix.diagonal() / matrix.sum(axis=1)
    emotions = [
        "anger",
        "surprise",
        "disgust",
        "joy",
        "fear",
        "sadness",
        "neutral",
    ]
    return {em: rec for em, rec in zip(emotions, avg_recalls)}


def get_confusion_matrix(
    predictions: np.ndarray, labels: np.ndarray
) -> pd.DataFrame:
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
    return confusion_matrix


def plot_emotion_comparison(all_data: dict[str, dict[str, float]]) -> None:
    dataframe = pd.DataFrame(columns=["Emotion", "Modality", "Recall"])
    for modality, emotion_values in all_data.items():
        for emotion, recall in emotion_values.items():
            new_data = pd.DataFrame.from_dict(
                {
                    "Emotion": [emotion],
                    "Modality": [modality],
                    "Recall": [recall],
                }
            )
            dataframe = pd.concat([dataframe, new_data])

    # Draw a nested barplot by species and sex
    g = sns.catplot(
        data=dataframe,
        kind="bar",
        x="Emotion",
        y="Recall",
        hue="Modality",
        palette="tab10",
        alpha=0.8,
        height=4,
        aspect=6.4 / 4,
    )
    g.set_axis_labels("Emotion", "Recall")
    g.legend.set_title("Data")
    plt.grid(axis="y")
    # plt.savefig("plots/emotion_comparison.pdf")
    plt.show()


def plot_confusion_matrix(predictions, labels, name):
    confusion_matrix = get_confusion_matrix(predictions, labels)
    plt.figure(figsize=(4, 3))
    sns.heatmap(confusion_matrix, annot=True, fmt=".2f", cmap="Blues")
    plt.xlabel("Predicted emotion")
    plt.ylabel("True emotion")
    plt.tight_layout()
    # plt.savefig(f"plots/{name}_confusion.pdf")
    plt.show()


def plot_confusion_matrix_difference(
    comp_predictions, comp_labels, train_predictions, train_labels, name
):
    comp_confusion = get_confusion_matrix(comp_predictions, comp_labels)
    train_confusion = get_confusion_matrix(train_predictions, train_labels)

    plt.figure(figsize=(4, 3))
    cmap = LinearSegmentedColormap.from_list("rg", ["r", "w", "g"], N=256)
    sns.heatmap(
        comp_confusion - train_confusion,
        annot=True,
        fmt=".2f",
        cmap=cmap,
        vmin=-1,
        vmax=1,
        center=0,
    )
    plt.xlabel("Predicted emotion")
    plt.ylabel("True emotion")
    plt.tight_layout()
    # plt.savefig(f"plots/{name}_confusion_difference.pdf")
    plt.show()


def main():
    os.makedirs("plots", exist_ok=True)
    matplotlib.rcParams.update({"font.size": 12})
    plt.rc("text", usetex=True)
    plt.rc("font", family="serif")

    all_data = {}

    # Text data
    classifier = ClassifierFactory.get("text", "distilbert")
    parameters = {"init_lr": 1e-05, "dropout_rate": 0.2, "dense_layer": 1024}
    save_path = "models/text/distilbert"
    parameters.update({"save_path": save_path})
    classifier.load(parameters)
    prediction = classifier.classify(parameters)
    labels = classifier.data_reader.get_labels(Set.TEST, parameters)
    all_data["Training"] = score_per_class(labels, prediction)

    classifier.data_reader = DataFactory.get_data_reader("comparison_text")
    cd_prediction = classifier.classify(parameters)
    cd_labels = classifier.data_reader.get_labels(Set.TEST, parameters)
    all_data["Comparison"] = score_per_class(cd_labels, cd_prediction)

    plot_confusion_matrix(cd_prediction, cd_labels, "text_comparison")
    plot_confusion_matrix_difference(
        cd_prediction, cd_labels, prediction, labels, "text_comparison"
    )

    classifier.data_reader = DataFactory.get_data_reader(
        "comparison_text", "data/comparison_dataset/text_sr"
    )
    sr_prediction = classifier.classify(parameters)
    sr_labels = classifier.data_reader.get_labels(Set.TEST, parameters)
    all_data["SR Comparison"] = score_per_class(sr_labels, sr_prediction)

    plot_confusion_matrix(sr_prediction, sr_labels, "text_sr_comparison")
    plot_confusion_matrix_difference(
        sr_prediction,
        sr_labels,
        cd_prediction,
        cd_labels,
        "text_sr_comparison",
    )

    classifier.data_reader = DataFactory.get_data_reader(
        "comparison_text", "data/comparison_dataset/text_deepspeech"
    )
    ds_prediction = classifier.classify(parameters)
    ds_labels = classifier.data_reader.get_labels(Set.TEST, parameters)
    all_data["DS Comparison"] = score_per_class(ds_labels, ds_prediction)

    plot_confusion_matrix(ds_prediction, ds_labels, "text_ds_comparison")
    plot_confusion_matrix_difference(
        ds_prediction,
        ds_labels,
        cd_prediction,
        cd_labels,
        "text_ds_comparison",
    )

    plot_emotion_comparison(all_data)


if __name__ == "__main__":
    main()
