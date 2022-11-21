""" Visualize how the different classifiers perform for different emotions. """
import copy
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
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
    plt.figure(figsize=(6.4, 4))

    for label, data in all_data.items():
        plt.plot(data.values(), list(range(1, 8)), label=label)

    plt.legend()
    plt.xlabel("Recall")
    plt.xlim([-0.01, 1.0])
    plt.yticks(list(range(1, 8)), labels=list(all_data.values())[0].keys())
    plt.ylabel("Emotion")
    plt.grid()
    plt.tight_layout()
    plt.savefig("plots/emotion_comparison.pdf")
    plt.show()


def main():
    os.makedirs("plots", exist_ok=True)
    matplotlib.rcParams.update({"font.size": 11})
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
    all_data["Text Comparison Dataset"] = score_per_class(labels, prediction)

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
    all_data["Image Comparison Dataset"] = score_per_class(labels, prediction)

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
    all_data["Speech Comparison Dataset"] = score_per_class(labels, prediction)

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
