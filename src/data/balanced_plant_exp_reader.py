""" This data reader reads the PlantSpikerBox data from the experiments. """

import copy
from typing import Dict, Tuple

import numpy as np
import tensorflow as tf

from src.data.data_reader import Set
from src.data.experiment_data_reader import ExperimentDataReader
from src.data.plant_exp_reader import PlantExperimentDataReader
from src.utils import reader_main


class BalancedPlantExperimentDataReader(ExperimentDataReader):
    """
    This data reader reads the plant spiker box files from the experiments
    and balances the classes exactly.
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
        super().__init__("balanced_plant_exp", folder or "data/plant")
        self.unbalanced_reader = PlantExperimentDataReader(
            folder, default_label_mode
        )
        self.sample_rate = 10_000

    def cleanup(self, parameters: Dict = None) -> None:
        """
        Function that cleans up the big data arrays for memory optimization.

        :param parameters: Parameter Dictionary
        """
        self.unbalanced_reader.cleanup(parameters)

    def get_seven_emotion_data(
        self, which_set: Set, batch_size: int = 64, parameters: Dict = None
    ) -> tf.data.Dataset:
        """
        Main data reading function which reads the CSV file into a dataset
        and also converts the emotion labels to the three emotion space.

        :param which_set: Which dataset to use - train, val or test
        :param batch_size: The batch size for the resulting dataset
        :param parameters: Additional arguments
        :return: The tensorflow Dataset instance
        """
        parameters = parameters or {}
        if parameters.get("balanced", False):
            return self._get_balanced_seven_emotion_data(
                which_set, batch_size, parameters
            )
        else:
            return self._get_unbalanced_seven_emotion_data(
                which_set, batch_size, parameters
            )

    def _get_balanced_seven_emotion_data(
        self, which_set: Set, batch_size: int = 64, parameters: Dict = None
    ) -> tf.data.Dataset:
        """
        Main data reading function which reads the plant data into a dataset.
        This function balances the different classes in the dataset.

        :param which_set: Which dataset to use - train, val or test
        :param batch_size: The batch size for the resulting dataset
        :param parameters: Additional parameters
        :return: The tensorflow Dataset instance
        """
        parameters = parameters or {}
        shuffle = parameters.get(
            "shuffle", True if which_set == Set.TRAIN else False
        )
        class_data = [
            np.empty((0, self.get_input_shape(parameters)[0]))
            for _ in range(7)
        ]
        class_datasets = []
        class_names = [
            "angry",
            "surprise",
            "disgust",
            "happy",
            "fear",
            "sad",
            "neutral",
        ]
        unb_parameters = copy.deepcopy(parameters)
        unb_parameters.update({"shuffle": False})
        for plant_data, labels in self._get_unbalanced_seven_emotion_data(
            which_set, 1024, unb_parameters
        ):
            plant_class = np.argmax(labels.numpy(), axis=1)
            plant_data = plant_data.numpy()
            for index in range(7):
                class_data[index] = np.concatenate(
                    [class_data[index], plant_data[plant_class == index, :]],
                    axis=0,
                )
        total_count = sum([cd.shape[0] for cd in class_data])
        for index, class_name in enumerate(class_names):
            labels = np.zeros((class_data[index].shape[0], 7))
            labels[:, index] = 1
            dataset = tf.data.Dataset.from_tensor_slices(
                (
                    tf.convert_to_tensor(class_data[index]),
                    tf.convert_to_tensor(labels),
                )
            )
            dataset = dataset.repeat()
            if shuffle:
                dataset = dataset.shuffle(1000)
            class_datasets.append(dataset)

        resampled_ds = tf.data.Dataset.sample_from_datasets(
            class_datasets, weights=[1 / 7.0] * 7
        )
        if shuffle:
            resampled_ds = resampled_ds.shuffle(1000)
        resampled_ds = resampled_ds.take(total_count).batch(
            batch_size=batch_size
        )
        return resampled_ds

    def get_three_emotion_data(
        self, which_set: Set, batch_size: int = 64, parameters: Dict = None
    ) -> tf.data.Dataset:
        """
        Main data reading function which reads the plant data into a dataset
        and also converts the emotion labels to the three emotion space.

        :param which_set: Which dataset to use - train, val or test
        :param batch_size: The batch size for the resulting dataset
        :param parameters: Additional arguments
        :return: The tensorflow Dataset instance
        """
        parameters = parameters or {}
        if parameters.get("balanced", False):
            raise NotImplementedError(
                "Balanced dataset not existing for three emotions!"
            )
        else:
            return self._get_unbalanced_three_emotion_data(
                which_set, batch_size, parameters
            )

    def _get_unbalanced_seven_emotion_data(
        self, which_set: Set, batch_size: int = 64, parameters: Dict = None
    ) -> tf.data.Dataset:
        """
        Get the unbalanced dataset from the PlantExperimentReader

        :param which_set: Which split of the data to use
        :param batch_size: Batch size for dataset
        :param parameters: Additional parameters
        :return: Dataset with unbalanced classes.
        """
        dataset = self.unbalanced_reader.get_seven_emotion_data(
            which_set, batch_size, parameters
        )
        return dataset

    def _get_unbalanced_three_emotion_data(
        self, which_set: Set, batch_size: int = 64, parameters: Dict = None
    ) -> tf.data.Dataset:
        """
        Create a dataset that uses only three emotions.

        :param which_set: Which set: Train, val or test
        :param batch_size: Batch size
        :param parameters: Additional parameters
        :return: Dataset with three emotion labels.
        """
        dataset = self.unbalanced_reader.get_three_emotion_data(
            which_set, batch_size, parameters
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
        dataset = self._get_unbalanced_seven_emotion_data(
            which_set, 100, parameters
        )
        labels = np.empty((0,))
        for _, batch_labels in dataset:
            labels = np.concatenate(
                [labels, np.argmax(batch_labels, axis=1)], axis=0
            )
        return labels

    def get_input_shape(self, parameters: Dict) -> Tuple[int]:
        """
        Returns the shape of a preprocessed sample.

        :param parameters: Parameter dictionary
        :return: Tuple that is the shape of the sample.
        """
        return self.unbalanced_reader.get_input_shape(parameters)


def _main():  # pragma: no cover
    reader = BalancedPlantExperimentDataReader()
    parameters = {
        "label_mode": "both",
        "cv_portions": 5,
        "balanced": True,
        "window": 20,
        "hop": 20,
    }
    for split in range(5):
        print(f"Split {split}/5")
        parameters["cv_split"] = split
        reader_main(reader, parameters)


if __name__ == "__main__":  # pragma: no cover
    _main()
