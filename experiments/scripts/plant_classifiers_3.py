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
    runner = CrossValidationExperimentRunner("plant_parameters_3")

    lstm_configs = make_dictionaries(
        {"epochs": 50, "patience": 10, "batch_size": 64},
        learning_rate=[0.0003],
        lstm_units=[512, 1024],
        downsampling_factor=[500, 1000, 2000],
        lstm_layers=[2],
        dropout=[0.2],
        label_mode=["expected", "faceapi", "both"],
        window=[10, 20],
        hop=[10],
    )
    lstm_configs_2 = lstm_configs.copy()
    for lstm_params in lstm_configs:
        lstm_params.update({"weighted": True})
    for lstm_params in lstm_configs_2:
        lstm_params.update({"balanced": True})
    lstm_configs.extend(lstm_configs_2)

    dense_configs = make_dictionaries(
        {"epochs": 50, "patience": 10, "batch_size": 64},
        learning_rate=[0.001],
        dense_units=[1024, 2048, 4096],
        downsampling_factor=[500, 1000, 2000],
        dense_layers=[2, 4],
        dropout=[0.2],
        label_mode=["expected", "faceapi", "both"],
        window=[10, 20],
        hop=[10],
    )
    dense_configs_2 = dense_configs.copy()
    for dense_params in dense_configs:
        dense_params.update({"weighted": True})
    for dense_params in dense_configs_2:
        dense_params.update({"balanced": True})
    dense_configs.extend(dense_configs_2)

    mfcc_configs = make_dictionaries(
        {"epochs": 50, "patience": 10, "batch_size": 64, "preprocess": False},
        learning_rate=[0.0003],
        conv_filters=[96],
        conv_layers=[2],
        conv_kernel_size=[7],
        dropout=[0, 0.2],
        label_mode=["expected", "faceapi", "both"],
        window=[10, 20],
        hop=[10],
    )
    mfcc_configs_2 = mfcc_configs.copy()
    for mfcc_params in mfcc_configs:
        mfcc_params.update({"weighted": True})
    for mfcc_params in mfcc_configs_2:
        mfcc_params.update({"balanced": True})
    mfcc_configs.extend(mfcc_configs_2)

    resnet_configs = make_dictionaries(
        {"epochs": 50, "patience": 10, "batch_size": 64, "preprocess": False},
        learning_rate=[0.0003, 0.001],
        dropout=[0, 0.2],
        label_mode=["expected", "faceapi", "both"],
        pretrained=[True, False],
        num_mfcc=[20, 40, 60],
        window=[10, 20],
        hop=[10],
    )
    resnet_configs_2 = resnet_configs.copy()
    for resnet_params in resnet_configs:
        resnet_params.update({"weighted": True})
    for resnet_params in resnet_configs_2:
        resnet_params.update({"balanced": True})
    resnet_configs.extend(resnet_configs_2)

    runner.add_grid_experiments(
        modality="plant",
        model="plant_lstm",
        train_parameters=lstm_configs,
    )

    runner.add_grid_experiments(
        modality="plant",
        model="plant_dense",
        train_parameters=dense_configs,
    )

    runner.add_grid_experiments(
        modality="plant",
        model="plant_mfcc_cnn",
        train_parameters=mfcc_configs,
    )

    runner.add_grid_experiments(
        modality="plant",
        model="plant_mfcc_resnet",
        train_parameters=resnet_configs,
    )

    my_experiments = list(range(len(runner.experiments)))[
        my_task_id::num_tasks
    ]

    print(
        f"Running {len(my_experiments)} out of "
        f"{len(runner.experiments)} experiments."
    )

    runner.run_all(indices=my_experiments)
