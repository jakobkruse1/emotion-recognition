"""This script runs the best image classifier configurations with augmentation
and class weighing and checks the impact of these changes."""
import sys

import tensorflow as tf

from src.experiment.experiment import ExperimentRunner, make_dictionaries

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

        else:
            tf.config.set_visible_devices(physical_devices[:1], "GPU")
            tf.config.experimental.set_memory_growth(
                physical_devices[0], False
            )


if __name__ == "__main__":
    best_efficientnet = [
        {
            "epochs": 50,
            "batch_size": 256,
            "patience": 15,
            "learning_rate": 0.001,
            "frozen_layers": 0,
            "extra_layer": 2048,
        },
        {
            "epochs": 50,
            "batch_size": 256,
            "patience": 15,
            "learning_rate": 0.003,
            "frozen_layers": 0,
            "extra_layer": 2048,
        },
        {
            "epochs": 50,
            "batch_size": 256,
            "patience": 15,
            "learning_rate": 0.001,
            "frozen_layers": 0,
            "extra_layer": 1024,
        },
        {
            "epochs": 50,
            "batch_size": 256,
            "patience": 15,
            "learning_rate": 0.003,
            "frozen_layers": 0,
            "extra_layer": 1024,
        },
        {
            "epochs": 50,
            "batch_size": 256,
            "patience": 15,
            "learning_rate": 0.001,
            "frozen_layers": 0,
            "extra_layer": 0,
        },
        {
            "epochs": 50,
            "batch_size": 256,
            "patience": 15,
            "learning_rate": 0.0003,
            "frozen_layers": 0,
            "extra_layer": 2048,
        },
        {
            "epochs": 50,
            "batch_size": 256,
            "patience": 15,
            "learning_rate": 0.0003,
            "frozen_layers": 0,
            "extra_layer": 1024,
        },
        {
            "epochs": 50,
            "batch_size": 256,
            "patience": 15,
            "learning_rate": 0.0001,
            "frozen_layers": 0,
            "extra_layer": 2048,
        },
        {
            "epochs": 50,
            "batch_size": 256,
            "patience": 15,
            "learning_rate": 0.0001,
            "frozen_layers": 0,
            "extra_layer": 1024,
        },
        {
            "epochs": 50,
            "batch_size": 256,
            "patience": 15,
            "learning_rate": 0.001,
            "frozen_layers": -200,
            "extra_layer": 1024,
        },
    ]

    best_vgg16 = [
        {
            "epochs": 30,
            "batch_size": 64,
            "patience": 8,
            "learning_rate": 0.0001,
            "deep": True,
            "dropout": 0.4,
            "frozen_layers": 0,
            "l1": 0,
            "l2": 1e-05,
        },
        {
            "epochs": 30,
            "batch_size": 64,
            "patience": 8,
            "learning_rate": 0.0001,
            "deep": False,
            "dropout": 0,
            "frozen_layers": 0,
            "l1": 0,
            "l2": 0.001,
        },
        {
            "epochs": 30,
            "batch_size": 64,
            "patience": 8,
            "learning_rate": 0.0001,
            "deep": True,
            "dropout": 0.2,
            "frozen_layers": 0,
            "l1": 0,
            "l2": 0.001,
        },
        {
            "epochs": 30,
            "batch_size": 64,
            "patience": 8,
            "learning_rate": 0.0001,
            "deep": True,
            "dropout": 0.4,
            "frozen_layers": 0,
            "l1": 0.0001,
            "l2": 0,
        },
        {
            "epochs": 30,
            "batch_size": 64,
            "patience": 8,
            "learning_rate": 0.0001,
            "deep": True,
            "dropout": 0.2,
            "frozen_layers": 0,
            "l1": 1e-05,
            "l2": 0.001,
        },
        {
            "epochs": 30,
            "batch_size": 64,
            "patience": 8,
            "learning_rate": 0.0001,
            "deep": False,
            "dropout": 0.4,
            "frozen_layers": 0,
            "l1": 0.0001,
            "l2": 0.0001,
        },
        {
            "epochs": 30,
            "batch_size": 64,
            "patience": 8,
            "learning_rate": 0.0001,
            "deep": True,
            "dropout": 0,
            "frozen_layers": 0,
            "l1": 0.0001,
            "l2": 0,
        },
        {
            "epochs": 30,
            "batch_size": 64,
            "patience": 8,
            "learning_rate": 0.0001,
            "deep": False,
            "dropout": 0,
            "frozen_layers": 0,
            "l1": 0,
            "l2": 1e-05,
        },
        {
            "epochs": 30,
            "batch_size": 64,
            "patience": 8,
            "learning_rate": 0.0001,
            "deep": True,
            "dropout": 0,
            "frozen_layers": 0,
            "l1": 0,
            "l2": 0.001,
        },
        {
            "epochs": 30,
            "batch_size": 64,
            "patience": 8,
            "learning_rate": 0.0001,
            "deep": True,
            "dropout": 0,
            "frozen_layers": 0,
            "l1": 0.001,
            "l2": 1e-05,
        },
    ]
    all_efficientnet = []
    for config in best_efficientnet:
        configs = make_dictionaries(
            config, augment=[True, False], weighted=[True, False]
        )
        all_efficientnet.extend(configs)

    all_vgg16 = []
    for config in best_vgg16:
        configs = make_dictionaries(
            config, augment=[True, False], weighted=[True, False]
        )
        all_vgg16.extend(configs)

    setup_gpu()
    # Bert grid search here
    runner = ExperimentRunner("image_augment_parameters")

    runner.add_grid_experiments(
        modality="image",
        model="efficientnet",
        train_parameters=all_efficientnet,
    )
    runner.add_grid_experiments(
        modality="image", model="vgg16", train_parameters=all_vgg16
    )

    my_experiments = list(range(len(runner.experiments)))[
        my_task_id::num_tasks
    ]
    runner.run_all(indices=my_experiments)
