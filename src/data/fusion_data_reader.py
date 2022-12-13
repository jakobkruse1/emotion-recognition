""" This data reader reads the fusion data from the experiments. """

from typing import Dict, Generator, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

from src.data.data_reader import Set
from src.data.experiment_data_reader import ExperimentDataReader
from src.utils import reader_main


class FusionProbDataReader(ExperimentDataReader):
    """
    This data reader reads fusion data from the experiments
    """

    def __init__(
        self,
        folder: str = None,
    ) -> None:
        """
        Initialize the fusion data reader for the experiment data.

        :param folder: The folder containing the data
        """
        super().__init__("fusion", folder or "data/continuous")
        self.default_label_mode = "expected"
        self.feature_sizes = {"image": 7, "plant": 7, "watch": 7}

    def get_raw_data(self, parameters: Dict) -> tuple[np.ndarray, np.ndarray]:
        """
        Function that reads all experiment emotion probabilities from the
        data/continuous folder.

        :param parameters: Parameters for the data reading process
        :return: Tuple with samples, labels
        """
        used_indices = self.get_complete_data_indices()
        modalities = parameters.get("modalities", ["image", "watch", "plant"])
        all_data = np.empty((0, *self.get_input_shape(parameters)))
        all_labels = np.empty((0,))
        exp_labels = np.zeros((613,))
        for emotion, times in self.emotion_times.items():
            exp_labels[
                int(times["start"]) : int(times["end"])
            ] = self.emotion_labels[emotion]
        for experiment_index in used_indices:
            data_path = f"data/continuous/{experiment_index:03d}_emotions.csv"
            df = pd.read_csv(data_path, index_col=0)
            col = 0
            exp_data = np.empty((613, len(modalities) * 7))
            for modality in modalities:
                for emotion in self.emotions:
                    exp_data[:, col] = df[f"{modality}_{emotion}"].values
                    col += 1
            all_data = np.concatenate([all_data, exp_data], axis=0)
            all_labels = np.concatenate([all_labels, exp_labels], axis=0)

        return all_data, all_labels

    def get_seven_emotion_data(
        self, which_set: Set, batch_size: int = 64, parameters: Dict = None
    ) -> tf.data.Dataset:
        """
        Method that returns a dataset of fusion probabilities.

        :param which_set: Which set to use.
        :param batch_size: Batch size for the dataset.
        :param parameters: Additional parameters.
        :return: Dataset instance.
        """
        parameters = parameters or {}

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
        all_data, all_labels = self.get_raw_data(parameters)
        set_data, set_labels = self.split_set(all_data, all_labels, which_set)

        def generator():
            for one_data, one_label in zip(set_data, set_labels):
                yield (
                    one_data,
                    tf.keras.utils.to_categorical(
                        np.array(one_label), num_classes=7
                    ),
                )

        return generator

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

    def get_input_shape(self, parameters: Dict) -> Tuple[int]:
        """
        Returns the shape of a concatenated input sample.

        :param parameters: Parameter dictionary
        :return: Tuple that is the shape of the sample.
        """
        parameters = parameters or {}
        shape = 0
        modalities = parameters.get("modalities", ["image", "watch", "plant"])
        for modality in modalities:
            shape += self.feature_sizes[modality]

        return (shape,)

    def split_set(
        self, all_data: np.ndarray, all_labels: np.ndarray, which_set: Set
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Split all labels into train, val and test sets.

        :param all_data: All data array shape (n_exp * 613, n_modalities * 7)
        :param all_labels: All corresponding labels (n_exp * 613,)
        :param which_set: Train, Val or Test set
        :return: Training, validation or test set as specified
        """
        x_train, x_med, y_train, y_med = train_test_split(
            all_data,
            all_labels,
            stratify=all_labels,
            test_size=0.4,
            random_state=42,
        )
        if which_set == Set.TRAIN:
            return x_train, y_train
        x_val, x_test, y_val, y_test = train_test_split(
            x_med, y_med, stratify=y_med, test_size=0.5, random_state=42
        )
        if which_set == Set.VAL:
            return x_val, y_val
        if which_set == Set.TEST:
            return x_test, y_test
        return all_data, all_labels


def _main():  # pragma: no cover
    reader = FusionProbDataReader()
    parameters = {}
    reader_main(reader, parameters)


if __name__ == "__main__":  # pragma: no cover
    _main()
