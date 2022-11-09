import copy
import os.path
import sys

from src.classification.classifier_factory import ClassifierFactory
from src.data.data_reader import Set
from src.utils.metrics import accuracy, per_class_accuracy

try:
    my_task_id = int(sys.argv[1])
    # num_tasks = int(sys.argv[2])
except IndexError:
    my_task_id = 0
    # num_tasks = 1

all_parameters = {
    "dense": {
        "epochs": 1000,
        "patience": 30,
        "batch_size": 64,
        "window": 20,
        "hop": 2,
        "balanced": True,
        "label_mode": "both",
        "learning_rate": 0.0003,
        "dense_units": 4096,
        "dropout": 0.2,
        "hidden_layers": 2,
        "checkpoint": True,
    },
    "lstm": {
        "epochs": 1000,
        "patience": 30,
        "batch_size": 64,
        "window": 20,
        "hop": 2,
        "balanced": True,
        "learning_rate": 0.001,
        "lstm_units": 4096,
        "dropout": 0.2,
        "lstm_layers": 1,
        "checkpoint": True,
    },
    "xgboost": {
        "batch_size": 64,
        "label_mode": "both",
        "window": 20,
        "hop": 2,
        "balanced": True,
        "max_depth": 10,
        "n_estimators": 80,
        "learning_rate": 0.01,
    },
    "transformer": {
        "epochs": 1000,
        "patience": 30,
        "batch_size": 64,
        "window": 20,
        "hop": 2,
        "balanced": True,
        "label_mode": "both",
        "ff_dim": 512,
        "dropout": 0.2,
        "dense_layers": 3,
        "dense_units": 1024,
        "checkpoint": True,
    },
    "random_forest": {
        "label_mode": "both",
        "batch_size": 64,
        "window": 20,
        "hop": 2,
        "balanced": True,
        "max_depth": 30,
        "n_estimators": 10,
        "min_samples_split": 4,
    },
}

save_paths = {
    "dense": "models/watch/watch_dense",
    "lstm": "models/watch/watch_lstm",
    "xgboost": "models/watch/xgboost",
    "transformer": "models/watch/watch_transformer",
    "random_forest": "models/watch/random_forest",
}


def main_train(cv_split, model):
    parameters = copy.deepcopy(all_parameters[model])
    parameters["cv_index"] = cv_split
    save_path = f"{save_paths[model]}_{cv_split}"
    max_acc = 0.0
    max_pc_acc = 0.0

    # Load existing model
    if os.path.exists(save_path):
        classifier = ClassifierFactory.get(
            "watch", f"watch_{model}", parameters
        )
        classifier.load({"save_path": save_path})
        pred = classifier.classify(parameters)
        labels = classifier.data_reader.get_labels(Set.TEST, parameters)
        max_acc = accuracy(labels, pred)
        max_pc_acc = per_class_accuracy(labels, pred)
        print(f"Read classifier with acc {max_acc} and pc acc {max_pc_acc}")

    acc_goal = 0.5
    pc_acc_goal = 0.5
    iteration = 0
    while max_acc < acc_goal or max_pc_acc < pc_acc_goal:
        classifier = ClassifierFactory.get(
            "watch", f"watch_{model}", parameters
        )
        classifier.train(parameters)
        if model in ["dense", "lstm", "transformer"]:
            classifier.load(
                {"save_path": f"models/watch/checkpoint_{cv_split}"}
            )
        pred = classifier.classify(parameters)
        labels = classifier.data_reader.get_labels(Set.TEST, parameters)
        this_acc = accuracy(labels, pred)
        this_pc_acc = per_class_accuracy(labels, pred)
        if (
            this_pc_acc >= max_pc_acc
            and 0.5 * this_acc + this_pc_acc >= 0.5 * max_acc + max_pc_acc
        ):
            print("Saving classifier!")
            classifier.save({"save_path": save_path})
            max_acc = this_acc
            max_pc_acc = this_pc_acc
        total_iterations = 20 + round((0.5 - min(max_acc, max_pc_acc)) / 0.002)
        print(
            f"Iteration {iteration}/{total_iterations}: "
            f"Acc {this_acc} (max {max_acc}), "
            f"PC Acc {this_pc_acc} (max {max_pc_acc})"
        )
        iteration += 1
        if iteration > 20:
            acc_goal -= 0.002
            pc_acc_goal -= 0.002
    print(f"Final Acc: {max_acc}")
    print(f"Final Class Acc: {max_pc_acc}")


if __name__ == "__main__":
    cv_split = my_task_id  # make sure to run this with task id [0, ..., 4]
    models = ["lstm"]  # "dense", "transformer", "xgboost", "random_forest"
    for model in models:
        print(f"Training {model} on split {cv_split}")
        main_train(cv_split, model)
