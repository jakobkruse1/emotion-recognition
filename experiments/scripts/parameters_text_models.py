"""Experiment that runs grid search over BERT parameters"""

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
    runner = ExperimentRunner("bert_models")
    train_parameters = make_dictionaries(
        init_lr=[1e-5, 3e-5], dropout_rate=[0.1, 0.2], dense_layer=[0, 512]
    )
    runner.add_grid_experiments(
        modality="text",
        model="bert",
        model_name=[
            "bert_en_uncased_L-2_H-128_A-2",
            "bert_en_uncased_L-4_H-128_A-2",
            "bert_en_uncased_L-4_H-256_A-4",
            "bert_en_uncased_L-2_H-256_A-4",
            "bert_en_uncased_L-6_H-256_A-4",
            "bert_en_uncased_L-4_H-512_A-8",
        ],
        train_parameters=train_parameters,
    )
    my_experiments = list(range(len(runner.experiments)))[
        my_task_id::num_tasks
    ]
    runner.run_all(indices=my_experiments)

    # Distilbert grid search here
    train_parameters = make_dictionaries(
        init_lr=[1e-5, 3e-5, 1e-4],
        dropout_rate=[0.1, 0.2],
        dense_layer=[0, 512, 1024],
    )
    runner = ExperimentRunner("distilbert_parameters")
    runner.add_grid_experiments(
        modality="text", model="distilbert", train_parameters=train_parameters
    )
    my_experiments = list(range(len(runner.experiments)))[
        my_task_id::num_tasks
    ]
    runner.run_all(indices=my_experiments)
