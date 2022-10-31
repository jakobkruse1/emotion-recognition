"""Experiment that runs grid search over VGG16 parameters"""

import sys

import tensorflow as tf

from src.experiment.experiment import ExperimentRunner, make_dictionaries

# HPC task distribution
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
    setup_gpu()
    # Bert grid search here
    runner = ExperimentRunner("vgg16_parameters")
    train_parameters = make_dictionaries(
        learning_rate=[1e-5, 1e-4, 1e-3],
        epochs=50,
        batch_size=64,
        patience=15,
        deep=[True, False],
        dropout=[0, 0.2, 0.4],
        frozen_layers=[0, -10, -1, -5],
        l1=[1e-3, 1e-4, 1e-5, 0],
        l2=[1e-3, 1e-4, 1e-5, 0],
    )
    runner.add_grid_experiments(
        modality="image",
        model="vgg16",
        train_parameters=train_parameters,
    )
    my_experiments = list(range(len(runner.experiments)))[
        my_task_id::num_tasks
    ]
    runner.run_all(indices=my_experiments)
