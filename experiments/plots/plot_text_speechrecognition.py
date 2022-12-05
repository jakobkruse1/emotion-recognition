""" Visualize how the text classifiers performs when using
speech recognition. """
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
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
        height=3,
        aspect=4.8 / 3,
    )
    g.set_axis_labels("Emotion", "Recall")
    g.legend.set_title("Text Data")
    plt.grid(axis="y")
    plt.savefig("plots/deepspeech_comparison.pdf")
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

    classifier.data_reader = DataFactory.get_data_reader(
        "comparison_text", "data/comparison_dataset/text_deepspeech"
    )
    ds_prediction = classifier.classify(parameters)
    ds_labels = classifier.data_reader.get_labels(Set.TEST, parameters)
    all_data["Deepspeech"] = score_per_class(ds_labels, ds_prediction)

    plot_emotion_comparison(all_data)


if __name__ == "__main__":
    main()
