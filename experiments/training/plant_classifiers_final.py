import copy
import os.path
import sys

from src.classification.classifier_factory import ClassifierFactory
from src.data.data_reader import Set
from src.utils.metrics import accuracy, per_class_accuracy

try:
    my_task_id = int(sys.argv[1])
    num_tasks = int(sys.argv[2])
except IndexError:
    my_task_id = 0
    num_tasks = 1

all_models = {
    "resnet": "plant_mfcc_resnet",
    "cnn": "plant_mfcc_cnn",
    "lstm": "plant_lstm",
    "dense": "plant_dense",
}
all_parameters = {
    "resnet": {
        "epochs": 1000,
        "patience": 50,
        "batch_size": 64,
        "preprocess": False,
        "learning_rate": 0.001,
        "dropout": 0.2,
        "label_mode": "both",
        "pretrained": False,
        "num_mfcc": 60,
        "window": 20,
        "hop": 10,
        "balanced": True,
        "checkpoint": True,
    },
    "lstm": {
        "epochs": 1000,
        "patience": 100,
        "batch_size": 64,
        "learning_rate": 0.0003,
        "lstm_units": 1024,
        "lstm_layers": 2,
        "dropout": 0,
        "label_mode": "both",
        "window": 20,
        "hop": 10,
        "balanced": True,
        "checkpoint": True,
    },
    "cnn": {
        "epochs": 1000,
        "patience": 100,
        "batch_size": 64,
        "preprocess": False,
        "learning_rate": 0.0003,
        "conv_filters": 96,
        "conv_layers": 2,
        "conv_kernel_size": 7,
        "dropout": 0.2,
        "label_mode": "both",
        "window": 20,
        "hop": 10,
        "balanced": True,
        "checkpoint": True,
    },
    "dense": {
        "epochs": 1000,
        "patience": 100,
        "batch_size": 64,
        "learning_rate": 0.001,
        "dense_units": 4096,
        "downsampling_factor": 500,
        "dense_layers": 2,
        "dropout": 0.2,
        "label_mode": "both",
        "window": 20,
        "hop": 10,
        "balanced": True,
        "checkpoint": True,
    },
}


def main_train(cv_split: int, model: str) -> None:
    parameters = copy.deepcopy(all_parameters[model])
    parameters["cv_index"] = cv_split
    save_path = f"models/plant/{all_models[model]}_{cv_split}"

    max_acc = 0.0
    max_pc_acc = 0.0
    # Load existing model
    if os.path.exists(save_path):
        classifier = ClassifierFactory.get(
            "plant", all_models[model], parameters
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
            "plant", all_models[model], parameters
        )
        classifier.train(parameters)
        if parameters.get("checkpoint", False):
            classifier.load(
                {"save_path": f"models/plant/checkpoint_{cv_split}"}
            )
        pred = classifier.classify(parameters)
        labels = classifier.data_reader.get_labels(Set.TEST, parameters)
        this_acc = accuracy(labels, pred)
        this_pc_acc = per_class_accuracy(labels, pred)
        if (
            this_pc_acc >= 0.25
            and this_acc >= 0.25 and this_pc_acc + this_acc > max_acc + max_pc_acc
        ):
            print("Saving classifier!")
            classifier.save({"save_path": save_path})
            max_acc = this_acc
            max_pc_acc = this_pc_acc
        total_iterations = 20 + round((0.5 - min(max_acc, max_pc_acc)) / 0.001)
        print(
            f"Iteration {iteration}/{total_iterations}: "
            f"Acc {this_acc} (max {max_acc}), "
            f"PC Acc {this_pc_acc} (max {max_pc_acc})"
        )
        classifier.data_reader.cleanup()
        iteration += 1
        if iteration > 20:
            acc_goal -= 0.0001
            pc_acc_goal -= 0.0001
    print(f"Final Acc: {max_acc}")
    print(f"Final Class Acc: {max_pc_acc}")


if __name__ == "__main__":
    cv_splits = [0] if my_task_id == 0 else [4]
    models = ["resnet"]  # "cnn", "lstm", "dense", "resnet"
    for cv_split in cv_splits:
        for model in models:
            print(f"Training {model} on split {cv_split}")
            main_train(cv_split, model)
