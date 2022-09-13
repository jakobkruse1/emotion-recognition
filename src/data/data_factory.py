"""This class implements a factory for easy access to data readers and data"""

from typing import Dict

import tensorflow as tf

from src.data.balanced_image_data_reader import BalancedImageDataReader
from src.data.balanced_plant_exp_reader import (
    BalancedPlantExperimentDataReader,
)
from src.data.data_reader import DataReader, Set
from src.data.image_data_reader import ImageDataReader
from src.data.plant_exp_reader import PlantExperimentDataReader
from src.data.speech_data_reader import SpeechDataReader
from src.data.text_data_reader import TextDataReader


class DataFactory:
    """
    The Data Factory returning data readers or data sets
    """

    @staticmethod
    def get_data_reader(data_type: str, data_folder=None) -> DataReader:
        """
        This factory method returns a data reader instance

        :param data_type: The type of data to return the reader for
        :param data_folder: Override data folder for the data reader
        :raise ValueError: If the data_type does not exist
        :return: A DataReader for the specified data type
        """
        if data_type == "text":
            return TextDataReader(folder=data_folder)
        elif data_type == "image":
            return ImageDataReader(folder=data_folder)
        elif data_type == "balanced_image":
            return BalancedImageDataReader(folder=data_folder)
        elif data_type == "speech":
            return SpeechDataReader(folder=data_folder)
        elif data_type == "plant":
            return PlantExperimentDataReader(folder=data_folder)
        elif data_type == "balanced_plant":
            return BalancedPlantExperimentDataReader(folder=data_folder)
        else:
            raise ValueError(
                f'The Data Reader for type "{data_type}" ' f"does not exist!"
            )

    @staticmethod
    def get_dataset(
        data_type: str,
        which_set: Set,
        emotions: str = "neutral_ekman",
        batch_size: int = 64,
        data_folder: str = None,
        parameters: Dict = None,
    ) -> tf.data.Dataset:
        """
        Get a specific dataset from a data reader

        :param data_type: The data type to consider
        :param which_set: Which dataset to return: train, val or test
        :param emotions: Which emotion set to use: neutral_ekman or three
        :param batch_size: The batch size for the returned dataset
        :param data_folder: The folder where data is stored
        :param parameters: Additional parameters for creating data
        :raise ValueError: If the emotion type is not available
        :return: Dataset instance that was requested
        """
        data_reader = DataFactory.get_data_reader(data_type, data_folder)
        if emotions == "neutral_ekman":
            return data_reader.get_seven_emotion_data(
                which_set, batch_size, parameters
            )
        elif emotions == "three":
            return data_reader.get_three_emotion_data(
                which_set, batch_size, parameters
            )
        else:
            raise ValueError(f'Emotion type "{emotions}" does not have data!')
