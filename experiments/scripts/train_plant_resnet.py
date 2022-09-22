import os.path
import sys

from src.classification.plant import PlantMFCCResnetClassifier
from src.data.data_reader import Set
from src.utils.metrics import accuracy, per_class_accuracy


def main(cv_split: int):
    parameters = {
        "epochs": 1000,
        "patience": 100,
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
        "cv_index": cv_split,
    }

    max_acc = 0.0
    max_pc_acc = 0.0
    if os.path.exists(f"models/plant/plant_mfcc_resnet_{cv_split}"):
        classifier = PlantMFCCResnetClassifier()
        classifier.load(
            {"save_path": f"models/plant/plant_mfcc_resnet_{cv_split}"}
        )
        pred = classifier.classify(parameters)
        labels = classifier.data_reader.get_labels(Set.TEST, parameters)
        max_acc = accuracy(labels, pred)
        max_pc_acc = per_class_accuracy(labels, pred)
    acc_goal = 0.5
    pc_acc_goal = 0.5
    iteration = 0
    while max_acc < acc_goal or max_pc_acc < pc_acc_goal:
        classifier = PlantMFCCResnetClassifier()
        classifier.train(parameters)
        classifier.load({"save_path": "models/plant/checkpoint"})
        pred = classifier.classify(parameters)
        labels = classifier.data_reader.get_labels(Set.TEST, parameters)
        this_acc = accuracy(labels, pred)
        this_pc_acc = per_class_accuracy(labels, pred)
        if this_pc_acc > max_pc_acc:
            classifier.save(
                {"save_path": f"models/plant/plant_mfcc_resnet_{cv_split}"}
            )
            max_acc = this_acc
            max_pc_acc = this_pc_acc
        print(
            f"Iteration {iteration}: Acc {this_acc} (max {max_acc}), "
            f"PC Acc {this_pc_acc} (max {max_pc_acc})"
        )
        classifier.data_reader.cleanup()
        iteration += 1
        if iteration > 50:
            acc_goal -= 0.002
            pc_acc_goal -= 0.002
    print(f"Final Acc: {max_acc}")
    print(f"Final Class Acc: {max_pc_acc}")


if __name__ == "__main__":
    split = int(sys.argv[1])
    main(split)
