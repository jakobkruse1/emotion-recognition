"""This script evaluates the overall best BYOL-S models."""
import glob

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from src.evaluation.evaluator import Evaluator

if __name__ == "__main__":  # pragma: no cover
    evaluator = Evaluator()
    evaluator.read_results(
        glob.glob("experiments/results/byols_parameters/*.json")
    )
    accuracies = evaluator.get_scores("accuracy")
    recalls = evaluator.get_scores("avg_recall")
    parameters = evaluator.get_parameters()

    for model in ["default", "cvt", "resnetish34"]:
        model_ids = []
        for index, param in enumerate(parameters):
            if param["train_parameters"]["model_name"] == model:
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
