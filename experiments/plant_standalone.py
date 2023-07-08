""" Run plant inference on a data file"""

import os
from typing import Generator

import numpy as np
import tensorflow as tf
from scipy.io import wavfile

from src.classification.plant import PlantMFCCResnetClassifier


def get_data_generator(
    data: np.ndarray, sample_rate: int, parameters: dict
) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
    """
    Generator that generates the data

    :param data: The plant data in an array
    :param sample_rate: The sample rate of the file
    :param parameters: Additional parameters including:
        - window: The length of the window to use in seconds
    :return: Generator that yields data and label.
    """
    window = parameters.get("window", 10)
    hop = parameters.get("hop", 5)

    def generator():
        for second in range(window, int(data.shape[0] / sample_rate), hop):
            sample = np.reshape(
                data[(second - window) * sample_rate : second * sample_rate],
                (-1,),
            )
            yield sample,

    return generator


def create_dataset(file_path: str, parameters: dict) -> tf.data.Dataset:
    """
    Function that creates a tensorflow dataset from a single plant file.

    :param file_path: The file path of the plant file
    :param parameters: Parameter dictionary
    :return: Tensorflow dataset with the plant data
    """
    sample_rate, data = wavfile.read(file_path)
    mean = np.mean(data)
    std = np.std(data)
    data = (data - mean) / std
    window = parameters.get("window", 10)
    batch_size = parameters.get("batch_size", 64)

    dataset = tf.data.Dataset.from_generator(
        get_data_generator(data, sample_rate, parameters),
        output_types=(tf.float32,),
        output_shapes=(tf.TensorShape([window * sample_rate]),),
    )
    dataset = dataset.batch(batch_size)
    return dataset


def run_inference(
    classifier: PlantMFCCResnetClassifier, dataset: tf.data.Dataset
) -> None:
    """
    Run inference on a plant dataset.

    :param classifier: The classifier to use for the prediction
    :param dataset: The dataset that contains the plant data in windows.
    """
    results = classifier.model.predict(dataset)
    emotion_ids = np.argmax(results, axis=1)
    print(results.shape)
    print(np.unique(emotion_ids, return_counts=True))

    # Order: "anger", "surprise", "disgust", "joy", "fear", "sadness", "neutral"


def main() -> None:
    """
    Main function that performs inference on a plant file
    """
    classifier = PlantMFCCResnetClassifier()
    parameters = {
        "batch_size": 64,
        "num_mfcc": 60,  # Do not change
        "window": 20,  # Do not change
        "hop": 3,
        "save_path": os.path.join("models", "plant", "plant_mfcc_resnet"),
    }
    classifier.load(parameters)

    dataset = create_dataset(
        os.path.join("data", "plant", "007_raw_spikerbox.wav"), parameters
    )
    run_inference(classifier, dataset)


if __name__ == "__main__":
    main()
