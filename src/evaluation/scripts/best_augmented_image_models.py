"""This script evaluates the overall best image models."""
import glob

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from src.evaluation.evaluator import Evaluator

if __name__ == "__main__":  # pragma: no cover
    evaluator = Evaluator()
    evaluator.read_results(
        glob.glob("experiments/results/image_augment_parameters/*.json")
        + glob.glob("experiments/results/image_balance_parameters/*.json")
    )
    accuracies = evaluator.get_scores("accuracy")
    recalls = evaluator.get_scores("avg_recall")
    parameters = evaluator.get_parameters()
    balanced = [
        "Weighted"
        if params["train_parameters"].get("weighted", False)
        else "Balanced"
        if params["train_parameters"].get("balanced", False)
        else "Unequal"
        for params in parameters
    ]
    augment = [
        "Augment" if params["train_parameters"]["augment"] else "No Augment"
        for params in parameters
    ]

    sns.scatterplot(
        x=accuracies,
        y=recalls,
        hue=augment,
        style=balanced,
        legend="full",
        s=50,
    )
    plt.xlabel("Accuracy")
    plt.ylabel("Recall")
    plt.title("Augmentation and Balancing Comparison")
    plt.xlim([0.4, 0.7])
    plt.ylim([0.4, 0.7])
    plt.grid()
    plt.show()

    models = [params["model"] for params in parameters]
    sns.scatterplot(
        x=accuracies,
        y=recalls,
        hue=models,
        style=balanced,
        legend="full",
        s=50,
    )
    plt.xlabel("Accuracy")
    plt.ylabel("Recall")
    plt.title("Models Comparison")
    plt.xlim([0.4, 0.7])
    plt.ylim([0.4, 0.7])
    plt.grid()
    plt.show()

    for model in ["efficientnet", "vgg16", "cross_attention"]:
        model_ids = []
        for index, param in enumerate(parameters):
            if param["model"] == model:
                model_ids.append(index)
        model_acc = np.asarray(accuracies)[model_ids]
        model_rec = np.asarray(recalls)[model_ids]
        highest_acc = np.max(model_acc)
        highest_rec = np.max(model_rec)
        highest_acc_id = model_ids[np.argmax(model_acc)]
        highest_rec_id = model_ids[np.argmax(model_rec)]
        if highest_rec_id == highest_acc_id:
            print(
                f"\n++++++\n{model} model with highest acc {highest_acc} and "
                f"rec {highest_rec} is: {parameters[highest_acc_id]}\n++++++\n"
            )
        else:
            print(
                f"\n++++++\n{model} has two different models with "
                f"highest acc and rec!"
            )
            print(
                f"Highest rec {highest_rec}, Acc {accuracies[highest_rec_id]},"
                f" Params: {parameters[highest_rec_id]}"
            )
            print(
                f"Highest acc {highest_acc}, Rec {recalls[highest_acc_id]},"
                f" Params: {parameters[highest_acc_id]}\n++++++\n"
            )
