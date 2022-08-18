""" This data reader reads the PlantSpikerBox data from the experiments. """

import glob
import os
from typing import Dict

import numpy as np
import tensorflow as tf

from src.data.data_reader import Set
from src.data.experiment_data_reader import ExperimentDataReader
from src.utils.ground_truth import experiment_ground_truth


class PlantExperimentDataReader(ExperimentDataReader):
    """
    This data reader reads the plant spiker box files from the experiments
    """

    def __init__(self, default_label_mode: str = "expected") -> None:
        """
        Initialize the plant data reader for the experiment data.

        :param default_label_mode: Whether to use expected emotion
            or face as ground truth.
        """
        super().__init__("plant_exp", "data/plant")
        self.default_label_mode = default_label_mode
        assert default_label_mode in ["expected", "faceapi"]
        self.files = glob.glob(os.path.join(self.folder, "*.wav"))
        if default_label_mode == "faceapi" and len(
            glob.glob("data/ground_truth/*.json")
        ) != len(self.files):
            self.prepare_faceapi_labels()

    def get_seven_emotion_data(
        self, which_set: Set, batch_size: int = 64, parameters: Dict = None
    ) -> tf.data.Dataset:
        parameters = parameters or {}
        _ = parameters.get("label_mode", self.default_label_mode)

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
        label_mode = parameters.get("label_mode", self.default_label_mode)
        if label_mode == "expected":
            return self.get_expected_labels()
        elif label_mode == "faceapi":
            return self.get_faceapi_labels()
        else:
            raise ValueError(
                f"Invalid label mode {label_mode}, "
                f"please use one of: 'expected', 'faceapi'"
            )

    def get_expected_labels(self) -> np.ndarray:
        pass

    def get_faceapi_labels(self) -> np.ndarray:
        pass

    @staticmethod
    def prepare_faceapi_labels() -> None:
        """
        This function prepares the faceapi labels if they are not computed yet.
        """
        for file in glob.glob("data/video/*.mp4"):
            emotions_file = os.path.join(
                "data",
                "ground_truth",
                f"{os.path.basename(file).split('.')[0]}_emotions.json",
            )
            if not os.path.exists(emotions_file):
                experiment_ground_truth(file)


if __name__ == "__main__":
    reader = PlantExperimentDataReader("faceapi")
