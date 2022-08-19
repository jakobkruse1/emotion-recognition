"""This script runs some plant classifier configs for testing"""
import sys

import tensorflow as tf

from src.experiment.cv_experiment import CrossValidationExperimentRunner
from src.experiment.experiment import make_dictionaries

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

    which_gpu = setup_gpu()
    # Start grid search with balanced data here
    runner = CrossValidationExperimentRunner("plant_parameters")

    lstm_configs = make_dictionaries(
        {"epochs": 1, "patience": 10, "batch_size": 8},
        learning_rate=[0.0003, 0.001],
        lstm_units=[64, 256, 1024],
        lstm_layers=[1, 2],
        dropout=[0, 0.2, 0.3],
        label_mode=["expected", "faceapi"],
        window=[5, 10, 20],
        hop=[5, 10],
        weighted=[False, True],
    )
    runner.add_grid_experiments(
        modality="plant",
        model="plant_lstm",
        train_parameters=lstm_configs,
    )

    my_experiments = list(range(len(runner.experiments)))[
        my_task_id::num_tasks
    ]

    print(
        f"Running {len(my_experiments)} out of "
        f"{len(runner.experiments)} experiments."
    )

    runner.run_all(indices=my_experiments)
