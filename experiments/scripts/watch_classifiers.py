"""This script runs some watch classifier configs for testing"""
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
    runner = CrossValidationExperimentRunner("watch_parameters")

    dense_configs = make_dictionaries(
        {"epochs": 200, "patience": 50, "batch_size": 64},
        label_mode=["both", "expected", "faceapi"],
        window=[5, 10, 20, 30],
        hop=[1, 2, 3, 4],
        balanced=[True, False],
        learning_rate=[0.0003, 0.001],
        dense_units=[512, 1024, 4096],
        dropout=[0, 0.2],
        hidden_layers=[2, 3],
    )
    runner.add_grid_experiments(
        modality="watch",
        model="watch_dense",
        train_parameters=dense_configs,
    )

    lstm_configs = make_dictionaries(
        {"epochs": 200, "patience": 50, "batch_size": 64},
        label_mode=["both", "expected", "faceapi"],
        window=[5, 10, 20, 30],
        hop=[1, 2, 3, 4],
        balanced=[True, False],
        learning_rate=[0.0003, 0.001],
        lstm_units=[512, 1024, 4096],
        dropout=[0, 0.2],
        lstm_layers=[1, 2],
    )
    runner.add_grid_experiments(
        modality="watch",
        model="watch_lstm",
        train_parameters=lstm_configs,
    )

    random_forest_configs = make_dictionaries(
        {"epochs": 200, "patience": 50, "batch_size": 64},
        label_mode=["both", "expected", "faceapi"],
        window=[5, 10, 20, 30],
        hop=[1, 2, 3, 4],
        balanced=[True, False],
        max_depth=[10, 30, 50],
        n_estimators=[10, 40, 80],
        min_samples_split=[2, 4, 8],
    )
    runner.add_grid_experiments(
        modality="watch",
        model="watch_random_forest",
        train_parameters=random_forest_configs,
    )

    transformer_configs = make_dictionaries(
        {"epochs": 200, "patience": 50, "batch_size": 64},
        label_mode=["both", "expected", "faceapi"],
        window=[5, 10, 20, 30],
        hop=[1, 2, 3, 4],
        balanced=[True, False],
        ff_dim=[512, 1024],
        dropout=[0, 0.2, 0.3],
        dense_layers=[2, 3],
        dense_units=[512, 1024, 2048],
    )
    runner.add_grid_experiments(
        modality="watch",
        model="watch_transformer",
        train_parameters=transformer_configs,
    )

    xgboost_configs = make_dictionaries(
        {"epochs": 200, "patience": 50, "batch_size": 64},
        label_mode=["both", "expected", "faceapi"],
        window=[5, 10, 20, 30],
        hop=[1, 2, 3, 4],
        balanced=[True, False],
        max_depth=[10, 30, 50],
        n_estimators=[10, 40, 80],
        learning_rate=[0.001, 0.01],
    )
    runner.add_grid_experiments(
        modality="watch",
        model="watch_xgboost",
        train_parameters=xgboost_configs,
    )

    my_experiments = list(range(len(runner.experiments)))[
        my_task_id::num_tasks
    ]

    print(len(dense_configs))
    print(len(lstm_configs))
    print(len(random_forest_configs))
    print(len(transformer_configs))
    print(len(xgboost_configs))

    print(
        f"Running {len(my_experiments)} out of "
        f"{len(runner.experiments)} experiments."
    )

    runner.run_all(indices=my_experiments[::-1])
