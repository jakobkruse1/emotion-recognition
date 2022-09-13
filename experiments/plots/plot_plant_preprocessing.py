""" Plot the plant data before and after preprocessing. """
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from src.classification.plant.plant_emotion_classifier import (
    PlantEmotionClassifier,
)
from src.data.plant_exp_reader import PlantExperimentDataReader, Set


def main():
    reader = PlantExperimentDataReader()
    ds_dataset = reader.get_seven_emotion_data(
        Set.TEST,
        1,
        {"window": 20, "hop": 10, "label_mode": "both", "preprocess": True},
    )
    rw_dataset = reader.get_seven_emotion_data(
        Set.TEST,
        1,
        {"window": 20, "hop": 10, "label_mode": "both", "preprocess": False},
    )
    for data, label in ds_dataset:
        ds_sample = np.reshape(data, (data.shape[1],))
        break
    for data, label in rw_dataset:
        rw_sample = np.reshape(data, (data.shape[1],))
        break

    plt.figure()
    plt.plot(np.linspace(0, 20, ds_sample.shape[0]), ds_sample)
    plt.xlabel("Plant Signal")
    plt.ylabel("Time in s")
    plt.show()

    plt.figure()
    plt.plot(np.linspace(0, 20, rw_sample.shape[0]), rw_sample)
    plt.xlabel("Plant Signal")
    plt.ylabel("Time in s")
    plt.show()

    mfcc_data = np.expand_dims(rw_sample, 0)
    mfcc_tensor = tf.convert_to_tensor(mfcc_data, dtype=tf.float32)
    mfccs = PlantEmotionClassifier.compute_mfccs(mfcc_tensor, {"num_mfcc": 60})
    mfccs = np.transpose(mfccs[0, :, :])
    plt.figure()
    plt.imshow(
        mfccs,
        cmap="inferno",
        norm=matplotlib.colors.Normalize(),
        aspect="auto",
    )
    plt.xlabel("Time")
    plt.ylabel("MFCC Number")
    plt.show()


if __name__ == "__main__":
    main()
