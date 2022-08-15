"""This script prints the five best image models from the experiments."""
import glob
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.data.data_factory import DataFactory, Set
from src.data.speech_data_reader import SpeechDataReader
from src.evaluation.evaluator import Evaluator

labels = DataFactory.get_data_reader("speech").get_labels(Set.TEST)


def plot_confusion_matrix(model_data, title="Confusion Matrix"):
    predictions = np.asarray(model_data["test_predictions"])
    if len(np.unique(predictions)) != 7:
        warnings.warn(
            f"{model_data['model']} did not produce all labels. Check model!"
        )
        return

    data = {"true": labels, "pred": predictions}
    df = pd.DataFrame(data, columns=["true", "pred"])
    confusion_matrix = pd.crosstab(
        df["true"], df["pred"], rownames=["True"], colnames=["Predicted"]
    )
    emotions = ["AN", "SU", "DI", "JO", "FE", "SA", "NE"]
    confusion_matrix.index = emotions
    confusion_matrix.columns = emotions
    sns.heatmap(confusion_matrix, annot=True, fmt="d")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.show()


def visualize_speech_data():
    dr = SpeechDataReader()
    dataset = dr.get_seven_emotion_data(Set.TEST, 1, {"shuffle": False})
    meld_data, crema_data = None, None
    meld_count = len(glob.glob("data/train/speech/test/**/*.wav"))
    for index, (audio, label) in enumerate(dataset):
        if index == 0:
            meld_data = audio.numpy()[0, :]
        if index == meld_count + 1:
            crema_data = audio.numpy()[0, :]
            break
    f, (ax1, ax2) = plt.subplots(2)
    ax1.plot(meld_data)
    ax1.set_title("MELD Audio Data")
    ax2.plot(crema_data)
    ax2.set_title("CREMA Audio Data")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":  # pragma: no cover
    # Best models
    evaluator = Evaluator()
    evaluator.read_results("experiments/results/speech_parameters/*.json")
    accuracies = evaluator.get_scores("accuracy")
    sorted_ind = np.argsort(-np.asarray(accuracies))
    sorted_acc = np.asarray(accuracies)[sorted_ind]
    parameters = evaluator.get_parameters()
    sorted_params = [parameters[ind] for ind in sorted_ind]

    print("++++++++ Best Models ++++++++")
    for i in range(2):
        print(f"Model {i+1}, Accuracy {sorted_acc[i]}")
        print(f"\tParameters: {sorted_params[i]}\n")

    best_model_id = sorted_ind[0]
    best_model_data = evaluator.result_data[best_model_id]
    plot_confusion_matrix(best_model_data, "Best model confusion")

    models = np.array([params["model"] for params in sorted_params])
    for model in np.unique(models):
        model_accs = sorted_acc[models == model]
        model_params = np.array(sorted_params)[models == model]
        model_ids = np.array(sorted_ind)[models == model]
        max_index = np.argmax(model_accs)
        print(
            f"{model} model max acc: {model_accs[max_index]}\n\t"
            f"Params: {model_params[max_index]}"
        )
        plot_confusion_matrix(
            evaluator.result_data[model_ids[max_index]],
            f"{model} confusion matrix",
        )

    # Class distribution
    classes = [
        "angry",
        "surprise",
        "disgust",
        "happy",
        "fear",
        "sad",
        "neutral",
    ]
    print("++++++++ Class Distribution ++++++++")
    for index, class_name in enumerate(classes):
        class_count = np.sum(labels == index)
        print(
            f"{class_name.ljust(10)}:\t {class_count} \t"
            f" {(class_count/labels.shape[0])*100:.1f}%"
        )

    # Visualize audio files
    visualize_speech_data()
