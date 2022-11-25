""" Visualize how the different classifiers perform for different emotions. """
import copy
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
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
    plt.savefig("plots/emotion_comparison.pdf")
    plt.show()


def plot_confusion_matrix(predictions, labels, title=None):
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
    # if title:
    #    plt.title(title)
    plt.tight_layout()
    plt.savefig(f"plots/{'_'.join(title.split(' '))}_confusion.pdf")
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
    all_data["Text Training"] = score_per_class(labels, prediction)

    classifier.data_reader = DataFactory.get_data_reader("comparison_text")
    prediction = classifier.classify(parameters)
    labels = classifier.data_reader.get_labels(Set.TEST, parameters)
    all_data["Text Comparison"] = score_per_class(labels, prediction)

    plot_confusion_matrix(prediction, labels, "Text Comparison")

    # Image Data
    classifier = ClassifierFactory.get("image", "vgg16")
    parameters = {"batch_size": 64, "deep": True}
    save_path = "models/image/vgg16"
    parameters.update({"save_path": save_path})
    classifier.load(parameters)
    prediction = classifier.classify(parameters)
    labels = classifier.data_reader.get_labels(Set.TEST, parameters)
    all_data["Image Training"] = score_per_class(labels, prediction)

    classifier.data_reader = DataFactory.get_data_reader("comparison_image")
    prediction = classifier.classify(parameters)
    labels = classifier.data_reader.get_labels(Set.TEST, parameters)
    all_data["Image Comparison"] = score_per_class(labels, prediction)

    plot_confusion_matrix(prediction, labels, "Image Comparison")

    # Speech Data
    classifier = ClassifierFactory.get("speech", "hubert")
    parameters = {"num_hidden_layers": 10, "extra_layer": 0, "batch_size": 32}
    save_path = "models/speech/hubert"
    parameters.update({"save_path": save_path})
    classifier.load(parameters)
    prediction = classifier.classify(parameters)
    labels = classifier.data_reader.get_labels(Set.TEST, parameters)
    all_data["Speech Training"] = score_per_class(labels, prediction)

    classifier.data_reader = DataFactory.get_data_reader("comparison_speech")
    prediction = classifier.classify(parameters)
    labels = classifier.data_reader.get_labels(Set.TEST, parameters)
    all_data["Speech Comparison"] = score_per_class(labels, prediction)

    plot_confusion_matrix(prediction, labels, "Speech Comparison")

    # Watch Data
    classifier = ClassifierFactory.get("watch", "watch_random_forest")
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
    all_data["Watch"] = score_per_class(labels, predictions)

    # Plant Data
    classifier = ClassifierFactory.get("plant", "plant_mfcc_resnet")
    parameters = {
        "batch_size": 64,
        "preprocess": False,
        "label_mode": "both",
        "pretrained": False,
        "num_mfcc": 60,
        "window": 20,
        "hop": 10,
    }
    save_path = "models/plant/plant_mfcc_resnet"
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
    all_data["Plant"] = score_per_class(labels, predictions)

    plot_emotion_comparison(all_data)


if __name__ == "__main__":
    tf.config.set_visible_devices([], "GPU")
    main()
