""" This data reader reads the PlantSpikerBox data from the experiments. """

import glob
import json
import os
from typing import Dict, Generator, List, Tuple

import numpy as np
import tensorflow as tf
from scipy.io import wavfile

from src.data.data_reader import Set
from src.data.experiment_data_reader import ExperimentDataReader
from src.utils.ground_truth import experiment_ground_truth


class PlantExperimentDataReader(ExperimentDataReader):
    """
    This data reader reads the plant spiker box files from the experiments
    """

    def __init__(
        self, folder: str = "data/plant", default_label_mode: str = "expected"
    ) -> None:
        """
        Initialize the plant data reader for the experiment data.

        :param folder: The folder that contains the plant files
        :param default_label_mode: Whether to use expected emotion
            or face as ground truth.
        """
        super().__init__("plant_exp", folder or "data/plant")
        self.default_label_mode = default_label_mode
        assert default_label_mode in ["expected", "faceapi"]
        self.files = glob.glob(os.path.join(self.folder, "*.wav"))
        self.files.sort()
        if default_label_mode == "faceapi" and len(
            glob.glob("data/ground_truth/*.json")
        ) != len(self.files):
            self.prepare_faceapi_labels()
        self.raw_data = []
        self.raw_labels = None
        self.sample_rate = 10_000

    def get_seven_emotion_data(
        self, which_set: Set, batch_size: int = 64, parameters: Dict = None
    ) -> tf.data.Dataset:
        parameters = parameters or {}
        label_mode = parameters.get("label_mode", self.default_label_mode)
        self.get_raw_data()
        self.get_raw_labels(label_mode)
        dataset = tf.data.Dataset.from_generator(
            self.get_data_generator(which_set, parameters),
            output_types=(tf.float32, tf.float32),
            output_shapes=(
                tf.TensorShape([self.get_input_shape(parameters)[0]]),
                tf.TensorShape([7]),
            ),
        )
        if parameters.get(
            "shuffle", True if which_set == Set.TRAIN else False
        ):
            dataset = dataset.shuffle(1024)
        dataset = dataset.batch(batch_size)
        return dataset

    def get_data_generator(
        self, which_set: Set, parameters: Dict
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generator that generates the data

        :param which_set: Train, val or test set
        :param parameters: Additional parameters including:
            - window: The length of the window to use in seconds
        :return: Generator that yields data and label.
        """

        def generator():
            window = parameters.get("window", 10)
            hop = parameters.get("hop", 5)
            preprocess = parameters.get("preprocess", True)
            indices = self.get_cross_validation_indices(which_set, parameters)
            for data_index in indices:
                data = self.raw_data[data_index]
                labels = self.raw_labels[data_index, :]
                for second in range(window, self.raw_labels.shape[1], hop):
                    sample = data[
                        (second - window)
                        * self.sample_rate : second
                        * self.sample_rate
                    ]
                    if preprocess:
                        sample = self.preprocess_sample(sample)
                    yield (
                        sample,
                        tf.keras.utils.to_categorical(
                            np.array(labels[second]), num_classes=7
                        ),
                    )

        return generator

    def get_cross_validation_indices(
        self, which_set: Set, parameters: Dict
    ) -> List[int]:
        """
        Generate a list of indices according to CrossValidation.

        :param which_set: Which set to use.
        :param parameters: Additional parameters including:
            - cv_portions: Number of cv splits to do.
            - cv_index: Which split to use.
        :return: List of indexes in a cv form.
        """
        if which_set == Set.ALL:
            return list(range(len(self.files)))
        cv_portions = parameters.get("cv_portions", 5)
        cv_index = parameters.get("cv_index", 0)
        assert cv_portions - 1 >= cv_index >= 0
        borders = np.linspace(0, len(self.files), cv_portions + 1).astype(int)
        if which_set == Set.TEST:
            test_split = cv_portions - cv_index
            return list(range(borders[test_split - 1], borders[test_split]))
        elif which_set == Set.VAL:
            val_split = (cv_portions - 1 - cv_index) % cv_portions
            val_split = val_split - 1 if val_split == 0 else val_split
            return list(range(borders[val_split - 1], borders[val_split]))
        elif which_set == Set.TRAIN:
            indices = []
            for i in range(1, cv_portions - 1):
                train_split = (i - cv_index) % cv_portions
                train_split = (
                    train_split - 1 if train_split == 0 else train_split
                )
                indices.extend(
                    list(range(borders[train_split - 1], borders[train_split]))
                )
            return indices

    def get_three_emotion_data(
        self, which_set: Set, batch_size: int = 64, parameters: Dict = None
    ) -> tf.data.Dataset:
        """
        Create a dataset that uses only three emotions.

        :param which_set: Which set: Train, val or test
        :param batch_size: Batch size
        :param parameters: Additional parameters
        :return: Dataset with three emotion labels.
        """
        dataset = self.get_seven_emotion_data(
            which_set, batch_size, parameters
        )
        dataset = dataset.map(
            lambda x, y: tf.numpy_function(
                func=self.map_emotions,
                inp=[x, y],
                Tout=(tf.float32, tf.float32),
            )
        )
        return dataset

    def get_labels(
        self, which_set: Set = Set.TRAIN, parameters: Dict = None
    ) -> np.ndarray:
        """
        This function returns labels for the dataset

        :param which_set: Which set to get the labels for.
        :param parameters: Additional parameters.
        :return: Label numpy array
        """
        parameters = parameters or {}
        parameters["shuffle"] = False
        dataset = self.get_seven_emotion_data(which_set, 100, parameters)
        labels = np.empty((0,))
        for _, batch_labels in dataset:
            labels = np.concatenate(
                [labels, np.argmax(batch_labels, axis=1)], axis=0
            )
        return labels

    def get_raw_labels(self, label_mode: str) -> None:
        """
        Get the raw labels per experiment and time.
        Populates the raw_labels member of this class.
        The two axis are [experiment_index, time_in_seconds]

        :param label_mode: Whether to use expected or faceapi labels
        """
        self.raw_labels = np.zeros((len(self.files), 690))
        if label_mode == "expected":
            self.get_raw_expected_labels()
        elif label_mode == "faceapi":
            self.get_raw_faceapi_labels()
        self.raw_labels = self.raw_labels[:, :613]

    def get_raw_expected_labels(self):
        """
        Load the raw emotions from the expected emotions during the video.
        """
        for emotion, times in self.emotion_times.items():
            self.raw_labels[
                :, int(times["start"]) : int(times["end"])
            ] = self.emotion_labels[emotion]

    def get_raw_faceapi_labels(self):
        """
        Load the raw labels from the faceapi output files.
        """
        emotions_sorted = [
            "angry",
            "surprised",
            "disgusted",
            "happy",
            "fearful",
            "sad",
            "neutral",
        ]
        for file_index, file in enumerate(self.files):
            experiment_index = os.path.basename(file)[:3]
            ground_truth_file = glob.glob(
                f"data/ground_truth/{experiment_index}*.json"
            )[0]
            with open(ground_truth_file, "r") as emotions_file:
                raw_emotions = json.load(emotions_file)
            previous = None
            for time, emotion_probs in raw_emotions:
                time_index = int(float(time)) - 1
                if emotion_probs != ["undefined"]:
                    emotion_probs_sorted = [
                        emotion_probs[0][emotion]
                        for emotion in emotions_sorted
                    ]
                    label = np.argmax(emotion_probs_sorted)
                    previous = label
                else:
                    label = previous
                self.raw_labels[file_index, time_index] = label

    @staticmethod
    def prepare_faceapi_labels() -> None:
        """
        This function prepares the faceapi labels if they are not computed yet.
        """
        video_files = glob.glob("data/video/*.mp4")
        video_files.sort()
        for file in video_files:
            emotions_file = os.path.join(
                "data",
                "ground_truth",
                f"{os.path.basename(file).split('.')[0]}_emotions.json",
            )
            if not os.path.exists(emotions_file):
                experiment_ground_truth(file)

    def get_raw_data(self) -> None:
        """
        Load the raw plant data from the wave files.
        """
        self.raw_data = []
        for index, plant_file in enumerate(self.files):
            sample_rate, data = wavfile.read(plant_file)
            assert sample_rate == 10000, "WAV file has incorrect sample rate!"
            mean = np.mean(data)
            var = np.var(data)
            data = (data - mean) / var
            self.raw_data.append(data)

    @staticmethod
    def preprocess_sample(
        sample: np.ndarray, parameters: Dict = None
    ) -> np.ndarray:
        """
        Gets a sample with shape (window_size * 10000,) and then preprocesses it
        before using it in the classifier.

        :param sample: The data sample to preprocess.
        :param parameters: Additional parameters for preprocessing.
        :return: The preprocessed sample.
        """
        downsampling_factor = 500
        pad_size = downsampling_factor - sample.shape[0] % downsampling_factor
        pad_size = 0 if pad_size == downsampling_factor else pad_size
        padded_sample = np.append(sample, np.zeros(pad_size) * np.NaN)
        downsampled_sample = np.nanmean(
            padded_sample.reshape(-1, downsampling_factor), axis=1
        )
        return downsampled_sample

    def get_input_shape(self, parameters: Dict) -> Tuple[int]:
        """
        Returns the shape of a preprocessed sample.

        :param parameters: Parameter dictionary
        :return: Tuple that is the shape of the sample.
        """
        parameters = parameters or {}
        window = parameters.get("window", 10)
        if not parameters.get("preprocess", True):
            return (window * self.sample_rate,)
        test_input = np.zeros((window * self.sample_rate,))
        test_sample = self.preprocess_sample(test_input)
        return test_sample.shape


if __name__ == "__main__":
    reader = PlantExperimentDataReader()
    reader.prepare_faceapi_labels()
    parameters = {"label_mode": "faceapi", "cv_portions": 5, "cv_index": 4}
    data = reader.get_seven_emotion_data(Set.VAL, 64, parameters).take(1)
    for batch, labels in data:
        print(batch.shape)
        print(labels.shape)
    print(f"Train size: {reader.get_labels(Set.TRAIN, parameters).shape[0]}")
    print(f"Val size: {reader.get_labels(Set.VAL, parameters).shape[0]}")
    print(f"Test size: {reader.get_labels(Set.TEST, parameters).shape[0]}")
    print(f"All size: {reader.get_labels(Set.ALL, parameters).shape[0]}")
