"""This plot shows the impact of balancing and augmentation."""
import glob
import os

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from src.evaluation.evaluator import Evaluator


def main():
    os.makedirs("plots", exist_ok=True)
    matplotlib.rcParams.update({"font.size": 12})
    plt.rc("text", usetex=True)
    plt.rc("font", family="serif")

    evaluator = Evaluator()
    evaluator.read_results(
        glob.glob("experiments/results/image_augment_parameters/*.json")
        + glob.glob("experiments/results/image_balance_parameters/*.json")
    )
    accuracies = evaluator.get_scores("accuracy")
    pc_accs = evaluator.get_scores("per_class_accuracy")
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

    plt.figure(figsize=(5.5, 3.5))
    sns.scatterplot(
        x=accuracies,
        y=pc_accs,
        hue=augment,
        style=balanced,
        legend="full",
        s=50,
    )
    plt.xlabel("Accuracy")
    plt.ylabel("Avg. Recall")
    plt.xlim([0.4, 0.65])
    plt.ylim([0.4, 0.65])
    plt.grid()
    plt.tight_layout()
    plt.savefig("plots/image_augmentation_balancing.pdf")
    plt.show()


if __name__ == "__main__":
    main()
