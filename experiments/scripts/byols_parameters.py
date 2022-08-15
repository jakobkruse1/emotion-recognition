"""This script runs some byols classifier configs for testing"""
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
    runner = ExperimentRunner("byols_parameters")

    byols_configs = make_dictionaries(
        {"epochs": 50, "patience": 10, "download": False},
        model_name=["cvt", "resnetish34", "default"],
        hidden=[1024, 2048, 4096, 512],
        learning_rate=[5e-5, 1e-4, 3e-4, 1e-3],
        freeze=[True, False],
        gpu=[which_gpu],
    )
    runner.add_grid_experiments(
        modality="speech",
        model="byols",
        train_parameters=byols_configs,
    )

    my_experiments = list(range(len(runner.experiments)))[
        my_task_id::num_tasks
    ]

    print(
        f"Running {len(my_experiments)} out of "
        f"{len(runner.experiments)} experiments."
    )

    runner.run_all(indices=my_experiments)
