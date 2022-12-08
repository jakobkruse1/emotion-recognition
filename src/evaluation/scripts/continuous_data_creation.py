""" This script is used to compare the plant, watch and image data
continuously. """
import copy
import json
import os

import numpy as np
import pandas as pd
import tensorflow as tf
from alive_progress import alive_bar
from scipy.io import wavfile

from src.classification.image import VGG16Classifier
from src.classification.plant import PlantMFCCResnetClassifier
from src.classification.watch import WatchRandomForestClassifier
from src.data.plant_exp_reader import PlantExperimentDataReader

INDICES = PlantExperimentDataReader.get_complete_data_indices()
EMOTIONS = [
    "anger",
    "surprise",
    "disgust",
    "joy",
    "fear",
    "sadness",
    "neutral",
]


def get_faceapi_groundtruth(experiment_index: int) -> np.ndarray:
    """
    Function extracts faceapi emotions from the data/ground_truth folder.

    :param experiment_index: Experiment index used to find correct experiment.
    :return: Numpy array of emotion probabilities for every second.
    """
    faceapi_emotions = [
        "angry",
        "surprised",
        "disgusted",
        "happy",
        "fearful",
        "sad",
        "neutral",
    ]
    labels = np.zeros((7, 613))
    filename = (
        f"data/ground_truth/{experiment_index:03d}_raw_video_emotions.json"
    )
    with open(filename, "r") as emotions_file:
        raw_emotions = json.load(emotions_file)
    previous = [1 / 7] * 7
    for time, emotion_probs in raw_emotions:
        time_index = int(float(time)) - 1
        if time_index > 612:
            continue
        if emotion_probs != ["undefined"]:
            label = [emotion_probs[0][emotion] for emotion in faceapi_emotions]
            previous = label
        else:
            label = previous

        labels[:, time_index] = label
    return labels


def get_plant_emotions(
    experiment_index: int, model_map: np.ndarray
) -> np.ndarray:
    """
    This function computes emotion probabilities from the plant data.

    :param experiment_index: The experiment index to compute emotions for.
    :param model_map: The array that maps elements to plant model.
        Output of the determine_correct_models function
    :return: Numpy array of shape (7, 690) giving emotion probabilities.
    """
    labels = np.zeros((7, 613))
    plant_file = f"data/plant/{experiment_index:03d}_raw_spikerbox.wav"
    sample_rate, data = wavfile.read(plant_file)
    assert sample_rate == 10000, "WAV file has incorrect sample rate!"
    mean = np.mean(data)
    std = np.std(data)
    data = (data - mean) / std
    samples = np.empty((0, 20 * sample_rate))
    for second in range(1, 614):
        sample = data[max(0, second - 20) * sample_rate : second * sample_rate]
        if sample.shape[0] != 20 * sample_rate:
            sample = np.concatenate(
                [
                    np.ones((20 * sample_rate - sample.shape[0],)) * sample[0],
                    sample,
                ],
                axis=0,
            )
        samples = np.concatenate([samples, sample.reshape((1, -1))], axis=0)
    # Samples is an array with shape (690, 200_000)
    which_models = model_map[INDICES.index(experiment_index), :]
    for model_id in range(5):
        model_samples = samples[which_models == model_id]
        model_labels = np.zeros((model_samples.shape[0],))
        if model_labels.shape[0] == 0:
            continue

        def generator():
            for sample, label in zip(model_samples, model_labels):
                yield (
                    sample,
                    tf.keras.utils.to_categorical(
                        np.array(label), num_classes=7
                    ),
                )

        dataset = tf.data.Dataset.from_generator(
            generator,
            output_types=(tf.float32, tf.float32),
            output_shapes=(tf.TensorShape([200_000]), tf.TensorShape([7])),
        )
        dataset = dataset.batch(64)
        model = PlantMFCCResnetClassifier()
        model.load({"save_path": f"models/plant/plant_mfcc_resnet_{model_id}"})
        results = model.model.predict(dataset, verbose=0)
        labels[:, which_models == model_id] = results.transpose()

    return labels


def get_watch_emotions(
    experiment_index: int, model_map: np.ndarray
) -> np.ndarray:
    """
    This function computes emotion probabilities from the watch data.

    :param experiment_index: The experiment index to compute emotions for.
    :param model_map: The array that maps elements to watch model.
        Output of the determine_correct_models function
    :return: Numpy array of shape (7, 690) giving emotion probabilities.
    """
    labels = np.zeros((7, 613))
    columns = [
        "HeartrateNorm",
        "AccelerometerXNorm",
        "AccelerometerYNorm",
        "AccelerometerZNorm",
        "AccelerometerNorm",
    ]
    raw_data = np.zeros((614, 5))
    for emotion in EMOTIONS:
        file = f"data/watch/{emotion}/{experiment_index:03d}_happimeter.csv"
        data = pd.read_csv(file, delimiter=",", usecols=columns)
        seconds = pd.read_csv(file, delimiter=",", usecols=["Second"])
        raw_data[seconds.values[:, 0], :] = data.values

    samples = np.zeros((613, 20, 5))
    for second in range(1, 614):
        sample = raw_data[max(0, second - 20) : second, :]
        if sample.shape[0] != 20:
            sample = np.concatenate(
                [np.ones((20 - sample.shape[0], 5)) * sample[0, :], sample],
                axis=0,
            )
        samples[second - 1, :, :] = sample

    which_models = model_map[INDICES.index(experiment_index), :]
    for model_id in range(5):
        model_samples = samples[which_models == model_id]
        if model_samples.shape[0] == 0:
            continue

        model_samples = np.reshape(model_samples, (model_samples.shape[0], -1))
        model = WatchRandomForestClassifier()
        model.load({"save_path": f"models/watch/random_forest_{model_id}"})
        results = model.model.predict_proba(model_samples)
        labels[:, which_models == model_id] = results.transpose()

    return labels


def add_to_dataframe(
    df: pd.DataFrame, to_add: np.ndarray, prefix: str
) -> pd.DataFrame:
    """
    This function adds a numpy array of shape (7, 690) to the dataframe using
    a prefix to distinguish data elements from each other.

    :param df: The dataframe to add data to.
    :param to_add: The numpy array to add to df, shape (7, 690)
    :param prefix: The prefix to use in column names
    :return: df with seven added columns
    """
    data_dict = {
        f"{prefix}_{emotion}": values
        for emotion, values in zip(EMOTIONS, to_add)
    }
    new_df = pd.concat(
        [df, pd.DataFrame.from_dict(data_dict, orient="columns")], axis=1
    )
    return new_df


def determine_correct_models(hop: int = 10) -> np.ndarray:
    """
    This function mimics the functionality of the plant data reader including
    cross validation splitting and both labels mode.
    It determines which model is used for which timestamp in all of the
    experiments. This is required to not use training data in the evaluation
    phase.

    :return: Numpy array of shape (54, 690) with integer values in the
        range [0, 4], indicating which of the 5 models to use at this timestamp.
    """
    dr = PlantExperimentDataReader()
    expected_labels = dr.get_raw_labels("expected")  # (54, 613)
    both_labels = dr.get_raw_labels("both")  # (54, 613) with -1 in wrong index

    file_indices = []
    timestamps = []
    raw_labels = []
    for index, experiment_index in enumerate(INDICES):
        for second in range(20, 613, hop):
            if both_labels[index, second] != -1:
                timestamps.append(second)
                file_indices.append(index)
                raw_labels.append(both_labels[index, second])
    raw_labels = np.asarray(raw_labels)

    emotion_borders = {}
    for emotion_index, emotion_name in zip(range(7), EMOTIONS):
        emotion_samples = np.where(raw_labels == emotion_index)[0]
        borders = np.linspace(0, emotion_samples.shape[0], 6).astype(int)
        emotion_borders[emotion_name] = [
            emotion_samples[border] for border in borders[:-1]
        ]

    right_models = np.zeros((54, 613))
    right_models[0, :19] = 4
    for index, experiment_index in enumerate(INDICES):
        for second in range(1, 614):
            if index == 0 and second < 20:
                continue
            current_emotion = expected_labels[index, second - 1]
            emotion_name = EMOTIONS[int(current_emotion)]
            borders = copy.copy(emotion_borders[emotion_name])
            borders[0] = 0
            borders.append(-1)
            split = -1
            for bor in range(len(borders) - 1):
                lower = borders[bor]
                upper = borders[bor + 1]
                lower_fulfilled = file_indices[lower] < index or (
                    file_indices[lower] == index
                    and timestamps[lower] <= second
                )
                upper_timestamp = 690 if index == 53 else timestamps[upper]
                upper_fulfilled = file_indices[upper] > index or (
                    file_indices[upper] == index and upper_timestamp > second
                )
                if lower_fulfilled and upper_fulfilled:
                    split = bor
            model = 4 - split
            right_models[index, second - 1] = model

    return right_models


def get_image_emotions(experiment_index: int) -> np.ndarray:
    """
    Function that creates all image emotion probabilities using our VGG16 model.

    :param experiment_index: The index of the experiment
    :return: Numpy array of shape (7, 613)
    """
    folder = f"data/continuous/face_images/{experiment_index:03d}"
    dataset = tf.keras.utils.image_dataset_from_directory(
        folder,
        shuffle=False,
        labels=None,
        batch_size=64,
        image_size=(224, 224),
        color_mode="grayscale",
    )
    fraction = 0.652
    dataset = dataset.map(
        lambda x: (
            tf.image.resize(tf.image.central_crop(x, fraction), (48, 48))
        )
    )
    dataset = dataset.map(lambda x: tf.image.grayscale_to_rgb(x))

    classifier = VGG16Classifier()
    classifier.load()
    results = classifier.model.predict(dataset)
    return results.transpose()


def obtain_emotion_probabilities(experiment_index: int) -> pd.DataFrame:
    """
    Function that obtains the emotion probabilities from all models and stores
    them in a pandas dataframe.

    :return: Dataframe that contains all emotion probabilities.
    """
    # Faceapi data
    dataframe = pd.DataFrame()
    faceapi_labels = get_faceapi_groundtruth(experiment_index)
    dataframe = add_to_dataframe(dataframe, faceapi_labels, "faceapi")

    # Plant data
    correct_models = determine_correct_models(hop=10)
    plant_labels = get_plant_emotions(experiment_index, correct_models)
    dataframe = add_to_dataframe(dataframe, plant_labels, "plant")

    # Smartwatch data
    correct_models = determine_correct_models(hop=2)
    watch_labels = get_watch_emotions(experiment_index, correct_models)
    dataframe = add_to_dataframe(dataframe, watch_labels, "watch")

    # Image data
    image_labels = get_image_emotions(experiment_index)
    dataframe = add_to_dataframe(dataframe, image_labels, "image")
    return dataframe


def main() -> None:
    """
    Main function that runs the evaluation of the continuous data.
    """
    os.makedirs("data/continuous", exist_ok=True)
    with alive_bar(54, title="Experiment", force_tty=True) as bar:
        for experiment_index in INDICES:
            experiment_data = obtain_emotion_probabilities(experiment_index)
            # Save to disk for later use
            experiment_data.to_csv(
                f"data/continuous/{experiment_index:03d}_emotions.csv"
            )
            bar()


if __name__ == "__main__":
    main()
