""" This data reader reads the PlantSpikerBox data from the experiments. """
import copy
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
        assert default_label_mode in ["expected", "faceapi", "both"]
        self.files = glob.glob(os.path.join(self.folder, "*.wav"))
        self.files.sort()
        if default_label_mode == "faceapi" and len(
            glob.glob("data/ground_truth/*.json")
        ) != len(self.files):
            self.prepare_faceapi_labels()
        self.raw_data = None
        self.raw_labels = None
        self.sample_rate = 10_000

    def cleanup(self, parameters: Dict = None) -> None:
        del self.raw_data
        del self.raw_labels

    def get_seven_emotion_data(
        self, which_set: Set, batch_size: int = 64, parameters: Dict = None
    ) -> tf.data.Dataset:
        parameters = parameters or {}
        self.get_raw_data(parameters)
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
            preprocess = parameters.get("preprocess", True)
            indices = self.get_cross_validation_indices(which_set, parameters)
            for data_index in indices:
                data = self.raw_data[data_index, :]
                label = self.raw_labels[data_index]
                if preprocess:
                    data = self.preprocess_sample(data)
                yield (
                    data,
                    tf.keras.utils.to_categorical(
                        np.array(label), num_classes=7
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
        cv_portions = parameters.get("cv_portions", 5)
        if which_set == Set.ALL:
            indices = []
            cv_params = copy.deepcopy(parameters)
            for cv_index in range(cv_portions):
                cv_params["cv_index"] = cv_index
                indices.extend(
                    self.get_cross_validation_indices(Set.TEST, cv_params)
                )
            return indices
        cv_index = parameters.get("cv_index", 0)
        assert cv_portions - 1 >= cv_index >= 0
        all_indices = []
        for emotion_index in range(7):
            emotion_samples = np.where(self.raw_labels == emotion_index)[0]
            borders = np.linspace(
                0, emotion_samples.shape[0], cv_portions + 1
            ).astype(int)
            if which_set == Set.TEST:
                test_split = cv_portions - cv_index
                all_indices.extend(
                    list(
                        emotion_samples[
                            borders[test_split - 1] : borders[test_split]
                        ]
                    )
                )
            elif which_set == Set.VAL:
                val_split = (cv_portions - 1 - cv_index) % cv_portions
                val_split = val_split - 1 if val_split == 0 else val_split
                all_indices.extend(
                    list(
                        emotion_samples[
                            borders[val_split - 1] : borders[val_split]
                        ]
                    )
                )
            elif which_set == Set.TRAIN:
                for i in range(1, cv_portions - 1):
                    train_split = (i - cv_index) % cv_portions
                    train_split = (
                        train_split - 1 if train_split == 0 else train_split
                    )
                    all_indices.extend(
                        list(
                            emotion_samples[
                                borders[train_split - 1] : borders[train_split]
                            ]
                        )
                    )
        all_indices.sort()
        return all_indices

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

    def get_raw_labels(self, label_mode: str) -> np.ndarray:
        """
        Get the raw labels per experiment and time.
        Populates the raw_labels member of this class.
        The two axis are [experiment_index, time_in_seconds]

        :param label_mode: Whether to use expected or faceapi labels
        :return: Array of all labels in shape (file, second)
        """
        raw_labels = np.zeros((len(self.files), 690))
        if label_mode == "expected":
            raw_labels = self.get_raw_expected_labels()
        elif label_mode == "faceapi":
            raw_labels = self.get_raw_faceapi_labels()
        elif label_mode == "both":
            expected = self.get_raw_expected_labels()
            faceapi = self.get_raw_faceapi_labels()
            expected[expected != faceapi] = -1
            raw_labels = expected
        return raw_labels[:, :613]

    def get_raw_expected_labels(self) -> np.ndarray:
        """
        Load the raw emotions from the expected emotions during the video.
        The expected emotion means that while the participant is watching a
        happy video, we expect them to be happy, thus the label is happy.

        :return: Labels that are expected from the user.
        """
        labels = np.zeros((len(self.files), 690))
        for emotion, times in self.emotion_times.items():
            labels[
                :, int(times["start"]) : int(times["end"])
            ] = self.emotion_labels[emotion]
        return labels

    def get_raw_faceapi_labels(self) -> np.ndarray:
        """
        Load the raw labels from the faceapi output files.

        :return: Labels that are collected from the user's face expression.
        """
        labels = np.zeros((len(self.files), 690))
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
                labels[file_index, time_index] = label
        return labels

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

    @staticmethod
    def _get_num_valid_data(
        all_labels: np.ndarray, parameters: Dict = None
    ) -> int:
        """
        Function that determines how many valid data entries do exist.
        This is used to allocate memory for the raw data arrays in advance.

        :param all_labels: The raw labels obtained from get_raw_labels
        :param parameters: Additional parameters
        :return: How many samples exist in Set.ALL
        """
        parameters = parameters or {}
        window = parameters.get("window", 10)
        hop = parameters.get("hop", 5)
        count = 0
        for file_index in range(all_labels.shape[0]):
            for second in range(window, all_labels.shape[1], hop):
                if 0 <= all_labels[file_index, second] < 7:
                    count += 1
        return count

    def get_raw_data(self, parameters: Dict) -> None:
        """
        Load the raw plant data from the wave files and split it into
        windows according to the parameters.

        :param parameters: Additional parameters
        """
        window = parameters.get("window", 10)
        hop = parameters.get("hop", 5)
        all_labels = self.get_raw_labels(
            parameters.get("label_mode", self.default_label_mode)
        )
        count = self._get_num_valid_data(all_labels, parameters)
        raw_data = np.empty((count, parameters.get("window", 10) * 10000))
        raw_labels = np.empty((count,))
        count = 0
        for index, plant_file in enumerate(self.files):
            sample_rate, data = wavfile.read(plant_file)
            assert sample_rate == 10000, "WAV file has incorrect sample rate!"
            mean = np.mean(data)
            var = np.var(data)
            data = (data - mean) / var
            labels = all_labels[index, :]
            for second in range(window, all_labels.shape[1], hop):
                if labels[second] == -1:
                    continue
                sample = np.reshape(
                    data[
                        (second - window)
                        * self.sample_rate : second
                        * self.sample_rate
                    ],
                    (1, -1),
                )
                raw_data[count, :] = sample
                raw_labels[count] = labels[second]
                count += 1
        self.raw_data = raw_data
        self.raw_labels = raw_labels

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
        parameters = parameters or {}
        downsampling_factor = parameters.get("downsampling_factor", 500)
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
    main_params = {
        "label_mode": "both",
        "cv_portions": 5,
        "window": 10,
        "hop": 10,
    }
    for cv_index in range(5):
        main_params["cv_index"] = cv_index
        main_data = reader.get_seven_emotion_data(Set.TRAIN, 64, main_params)
        main_all_labels = np.empty((0,))
        for _, mlabels in main_data:
            main_all_labels = np.concatenate(
                [main_all_labels, np.argmax(mlabels, axis=1)], axis=0
            )
        print(
            f"CV Split {cv_index}: Data Distribution "
            f"{np.unique(main_all_labels, return_counts=True)}"
        )
    print(f"Train size: {reader.get_labels(Set.TRAIN, main_params).shape[0]}")
    print(f"Val size: {reader.get_labels(Set.VAL, main_params).shape[0]}")
    print(f"Test size: {reader.get_labels(Set.TEST, main_params).shape[0]}")
    print(f"All size: {reader.get_labels(Set.ALL, main_params).shape[0]}")
