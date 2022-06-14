"""This script runs some speech classifier configs for testing"""
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
    runner = ExperimentRunner("speech_parameters")

    lstm_configs = make_dictionaries(
        {"epochs": 50, "patience": 10},
        learning_rate=[0.0003, 0.001, 0.003],
        lstm_units=[128, 256, 512],
        dropout=[0.2, 0.3],
        weighted=[True, False],
    )
    runner.add_grid_experiments(
        modality="speech",
        model="mfcc_lstm",
        train_parameters=lstm_configs,
    )

    hubert_configs = make_dictionaries(
        {"epochs": 50, "patience": 10},
        learning_rate=[5e-5, 0.0001, 0.0003],
        dropout=[0.1, 0.2],
        num_hidden_layers=[12, 10, 8],
        freeze=[True, False],
        extra_layer=[0, 1024],
        gpu=[which_gpu],
    )
    runner.add_grid_experiments(
        modality="speech",
        model="hubert",
        train_parameters=hubert_configs,
    )

    wav2vec2_configs = make_dictionaries(
        {"epochs": 50, "patience": 10},
        learning_rate=[5e-5, 0.0001, 0.0003],
        dropout=[0.1, 0.2],
        num_hidden_layers=[12, 10, 8],
        freeze=[True, False],
        extra_layer=[0, 1024],
        gpu=[which_gpu],
    )
    runner.add_grid_experiments(
        modality="speech",
        model="wav2vec2",
        train_parameters=wav2vec2_configs,
    )

    hmm_configs = make_dictionaries(
        {}, n_components=[4, 8, 12, 16], mfcc_num=[13, 20, 40]
    )
    runner.add_grid_experiments(
        modality="speech",
        model="hmm",
        train_parameters=hmm_configs,
    )

    runner.add_grid_experiments(
        modality="speech",
        model="gmm",
        train_parameters=hmm_configs,
    )

    svm_configs = make_dictionaries(
        {},
        mfcc_num=[13, 20, 40],
        kernel=["linear", "poly", "rbf", "sigmoid", "precomputed"],
    )
    runner.add_grid_experiments(
        modality="speech",
        model="svm",
        train_parameters=svm_configs,
    )

    byols_configs = make_dictionaries(
        {"epochs": 50, "patience": 10},
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
    runner.run_all(indices=my_experiments)
