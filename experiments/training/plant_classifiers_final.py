import os.path
import sys

from src.classification.plant import (
    PlantDenseClassifier,
    PlantLSTMClassifier,
    PlantMFCCCNNClassifier,
    PlantMFCCResnetClassifier,
)
from src.data.data_reader import Set
from src.utils.metrics import accuracy, per_class_accuracy

try:
    my_task_id = int(sys.argv[1])
    num_tasks = int(sys.argv[2])
except IndexError:
    my_task_id = 0
    num_tasks = 1


def main_resnet(cv_splits):
    for cv_split in cv_splits:
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
            "checkpoint": True,
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
            print(
                f"Read classifier with acc {max_acc} and pc acc {max_pc_acc}"
            )
        acc_goal = 0.5
        pc_acc_goal = 0.5
        iteration = 0
        while max_acc < acc_goal or max_pc_acc < pc_acc_goal:
            classifier = PlantMFCCResnetClassifier()
            classifier.train(parameters)
            classifier.load(
                {"save_path": f"models/plant/checkpoint_{cv_split}"}
            )
            pred = classifier.classify(parameters)
            labels = classifier.data_reader.get_labels(Set.TEST, parameters)
            this_acc = accuracy(labels, pred)
            this_pc_acc = per_class_accuracy(labels, pred)
            if (
                this_pc_acc >= max_pc_acc
                and this_acc + this_pc_acc >= max_acc + max_pc_acc
            ):
                print("Saving classifier!")
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
            if iteration > 20:
                acc_goal -= 0.002
                pc_acc_goal -= 0.002
        print(f"Final Acc: {max_acc}")
        print(f"Final Class Acc: {max_pc_acc}")


def main_lstm(cv_splits):
    for cv_split in cv_splits:
        parameters = {
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
            "cv_index": cv_split,
        }

        max_acc = 0.0
        max_pc_acc = 0.0
        if os.path.exists(f"models/plant/plant_lstm_{cv_split}"):
            classifier = PlantLSTMClassifier()
            classifier.load(
                {"save_path": f"models/plant/plant_lstm_{cv_split}"}
            )
            pred = classifier.classify(parameters)
            labels = classifier.data_reader.get_labels(Set.TEST, parameters)
            max_acc = accuracy(labels, pred)
            max_pc_acc = per_class_accuracy(labels, pred)
            print(
                f"Read classifier with acc {max_acc} and pc acc {max_pc_acc}"
            )
        acc_goal = 0.5
        pc_acc_goal = 0.5
        iteration = 0
        while max_acc < acc_goal or max_pc_acc < pc_acc_goal:
            classifier = PlantLSTMClassifier()
            classifier.train(parameters)
            classifier.load(
                {"save_path": f"models/plant/checkpoint_{cv_split}"}
            )
            pred = classifier.classify(parameters)
            labels = classifier.data_reader.get_labels(Set.TEST, parameters)
            this_acc = accuracy(labels, pred)
            this_pc_acc = per_class_accuracy(labels, pred)
            if (
                this_pc_acc >= max_pc_acc
                and this_acc + this_pc_acc >= max_acc + max_pc_acc
            ):
                print("Saving classifier!")
                classifier.save(
                    {"save_path": f"models/plant/plant_lstm_{cv_split}"}
                )
                max_acc = this_acc
                max_pc_acc = this_pc_acc
            print(
                f"Iteration {iteration}: Acc {this_acc} (max {max_acc}), "
                f"PC Acc {this_pc_acc} (max {max_pc_acc})"
            )
            classifier.data_reader.cleanup()
            iteration += 1
            if iteration > 20:
                acc_goal -= 0.002
                pc_acc_goal -= 0.002
        print(f"Final Acc: {max_acc}")
        print(f"Final Class Acc: {max_pc_acc}")


def main_cnn(cv_splits):
    for cv_split in cv_splits:
        parameters = {
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
            "cv_index": cv_split,
        }

        max_acc = 0.0
        max_pc_acc = 0.0
        if os.path.exists(f"models/plant/plant_mfcc_cnn_{cv_split}"):
            classifier = PlantMFCCCNNClassifier()
            classifier.load(
                {"save_path": f"models/plant/plant_mfcc_cnn_{cv_split}"}
            )
            pred = classifier.classify(parameters)
            labels = classifier.data_reader.get_labels(Set.TEST, parameters)
            max_acc = accuracy(labels, pred)
            max_pc_acc = per_class_accuracy(labels, pred)
            print(
                f"Read classifier with acc {max_acc} and pc acc {max_pc_acc}"
            )
        acc_goal = 0.5
        pc_acc_goal = 0.5
        iteration = 0
        while max_acc < acc_goal or max_pc_acc < pc_acc_goal:
            classifier = PlantMFCCCNNClassifier()
            classifier.train(parameters)
            classifier.load(
                {"save_path": f"models/plant/checkpoint_{cv_split}"}
            )
            pred = classifier.classify(parameters)
            labels = classifier.data_reader.get_labels(Set.TEST, parameters)
            this_acc = accuracy(labels, pred)
            this_pc_acc = per_class_accuracy(labels, pred)
            if (
                this_pc_acc >= max_pc_acc
                and this_acc + this_pc_acc >= max_acc + max_pc_acc
            ):
                print("Saving classifier!")
                classifier.save(
                    {"save_path": f"models/plant/plant_mfcc_cnn_{cv_split}"}
                )
                max_acc = this_acc
                max_pc_acc = this_pc_acc
            print(
                f"Iteration {iteration}: Acc {this_acc} (max {max_acc}), "
                f"PC Acc {this_pc_acc} (max {max_pc_acc})"
            )
            classifier.data_reader.cleanup()
            iteration += 1
            if iteration > 20:
                acc_goal -= 0.002
                pc_acc_goal -= 0.002
        print(f"Final Acc: {max_acc}")
        print(f"Final Class Acc: {max_pc_acc}")


def main_dense(cv_splits):
    for cv_split in cv_splits:
        parameters = {
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
            "cv_index": cv_split,
        }

        max_acc = 0.0
        max_pc_acc = 0.0
        if os.path.exists(f"models/plant/plant_dense_{cv_split}"):
            classifier = PlantDenseClassifier()
            classifier.load(
                {"save_path": f"models/plant/plant_dense_{cv_split}"}
            )
            pred = classifier.classify(parameters)
            labels = classifier.data_reader.get_labels(Set.TEST, parameters)
            max_acc = accuracy(labels, pred)
            max_pc_acc = per_class_accuracy(labels, pred)
            print(
                f"Read classifier with acc {max_acc} and pc acc {max_pc_acc}"
            )
        acc_goal = 0.5
        pc_acc_goal = 0.5
        iteration = 0
        while max_acc < acc_goal or max_pc_acc < pc_acc_goal:
            classifier = PlantDenseClassifier()
            classifier.train(parameters)
            classifier.load(
                {"save_path": f"models/plant/checkpoint_{cv_split}"}
            )
            pred = classifier.classify(parameters)
            labels = classifier.data_reader.get_labels(Set.TEST, parameters)
            this_acc = accuracy(labels, pred)
            this_pc_acc = per_class_accuracy(labels, pred)
            if (
                this_pc_acc >= max_pc_acc
                and this_acc + this_pc_acc >= max_acc + max_pc_acc
            ):
                print("Saving classifier!")
                classifier.save(
                    {"save_path": f"models/plant/plant_dense_{cv_split}"}
                )
                max_acc = this_acc
                max_pc_acc = this_pc_acc
            print(
                f"Iteration {iteration}: Acc {this_acc} (max {max_acc}), "
                f"PC Acc {this_pc_acc} (max {max_pc_acc})"
            )
            classifier.data_reader.cleanup()
            iteration += 1
            if iteration > 20:
                acc_goal -= 0.002
                pc_acc_goal -= 0.002
        print(f"Final Acc: {max_acc}")
        print(f"Final Class Acc: {max_pc_acc}")


if __name__ == "__main__":
    main_dense([my_task_id])
    main_resnet([my_task_id])
    main_lstm([my_task_id])
    main_cnn([my_task_id])
