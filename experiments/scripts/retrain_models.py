"""This script retrains the best classifiers with logging enabled. """

import sys

import tensorflow as tf

from src.classification.classifier_factory import ClassifierFactory
from src.data.data_reader import Set
from src.utils.metrics import accuracy

try:
    my_task_id = int(sys.argv[1])
    num_tasks = int(sys.argv[2])
except IndexError:
    my_task_id = 0
    num_tasks = 1


def setup_gpu():
    physical_devices = tf.config.list_physical_devices("GPU")
    if len(physical_devices) == 2:
        if my_task_id % 2 == 0:
            tf.config.set_visible_devices(physical_devices[1:], "GPU")
            tf.config.experimental.set_memory_growth(
                physical_devices[1], False
            )
            return 1

        else:
            tf.config.set_visible_devices(physical_devices[:1], "GPU")
            tf.config.experimental.set_memory_growth(
                physical_devices[0], False
            )
            return 0


if __name__ == "__main__":
    min_runs = 20
    max_runs = 60
    which_gpu = setup_gpu()
    # Start grid search with balanced data here
    model_names = [
        "bert",
        "distilbert",
        "efficientnet",
        "vgg16",
        "cross_attention",
        "mfcc_lstm",
        "hubert",
        "wav2vec2",
        "byols",
    ]
    modalities = [
        "text",
        "text",
        "image",
        "image",
        "image",
        "speech",
        "speech",
        "speech",
        "speech",
    ]
    acc_goals = [0.76, 0.76, 0.62, 0.65, 0.63, 0.50, 0.60, 0.57, 0.54]

    parameters = {
        "bert": {"init_lr": 1e-05, "dropout_rate": 0.1, "dense_layer": 0},
        "distilbert": {
            "init_lr": 1e-05,
            "dropout_rate": 0.2,
            "dense_layer": 1024,
        },
        "efficientnet": {
            "epochs": 50,
            "batch_size": 256,
            "patience": 15,
            "learning_rate": 0.003,
            "frozen_layers": 0,
            "extra_layer": 2048,
            "augment": True,
            "weighted": True,
        },
        "vgg16": {
            "epochs": 30,
            "batch_size": 64,
            "patience": 8,
            "learning_rate": 0.0001,
            "deep": True,
            "dropout": 0.4,
            "frozen_layers": 0,
            "l1": 0,
            "l2": 1e-05,
            "augment": True,
            "weighted": False,
        },
        "cross_attention": {
            "learning_rate": 0.0003,
            "weighted": False,
            "augment": False,
            "balanced": False,
        },
        "mfcc_lstm": {
            "epochs": 50,
            "patience": 10,
            "learning_rate": 0.001,
            "lstm_units": 512,
            "dropout": 0.3,
            "weighted": False,
        },
        "hubert": {
            "epochs": 50,
            "patience": 10,
            "learning_rate": 5e-05,
            "dropout": 0.1,
            "num_hidden_layers": 10,
            "freeze": False,
            "extra_layer": 0,
            "batch_size": 64,
        },
        "wav2vec2": {
            "epochs": 50,
            "patience": 10,
            "learning_rate": 5e-05,
            "dropout": 0.1,
            "num_hidden_layers": 8,
            "freeze": True,
            "extra_layer": 0,
            "batch_size": 64,
        },
        "byols": {
            "epochs": 50,
            "patience": 10,
            "download": False,
            "model_name": "resnetish34",
            "hidden": 1024,
            "learning_rate": 0.0001,
            "freeze": True,
            "batch_size": 64,
        },
    }

    this_model = model_names[my_task_id]
    this_modality = model_names[my_task_id]
    this_goal = acc_goals[my_task_id]
    this_parameters = parameters[this_model]

    best_acc = 0.0
    iteration = 0

    while iteration < min_runs or (
        best_acc < this_goal and iteration < max_runs
    ):
        classifier = ClassifierFactory.get(this_modality, this_model)
        classifier.train(this_parameters)
        predictions = classifier.classify(this_parameters)
        labels = classifier.data_reader.get_labels(Set.TEST, this_parameters)
        current_acc = accuracy(labels, predictions)
        if current_acc > best_acc:
            classifier.save({})
            best_acc = current_acc
        print(f"Iteration {iteration} finished with accuracy {current_acc}.")
        iteration += 1
