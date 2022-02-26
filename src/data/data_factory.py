"""This class implements a factory for easy access to data readers and data"""

import tensorflow as tf

from src.data.data_reader import DataReader, Set
from src.data.text_data_reader import TextDataReader


class DataFactory:
    """
    The Data Factory returning data readers or data sets
    """

    @staticmethod
    def get_data_reader(data_type: str) -> DataReader:
        """
        This factory method returns a data reader instance

        :param data_type: The type of data to return the reader for
        :raise ValueError: If the data_type does not exist
        :return: A DataReader for the specified data type
        """
        if data_type == "text":
            return TextDataReader()
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
    ) -> tf.data.Dataset:
        """
        Get a specific dataset from a data reader

        :param data_type: The data type to consider
        :param which_set: Which dataset to return: train, val or test
        :param emotions: Which emotion set to use: neutral_ekman or three
        :param batch_size: The batch size for the returned dataset
        :raise ValueError: If the emotion type is not available
        :return: Dataset instance that was requested
        """
        data_reader = DataFactory.get_data_reader(data_type)
        if emotions == "neutral_ekman":
            return data_reader.get_data(which_set, batch_size)
        elif emotions == "three":
            return data_reader.get_three_emotion_data(which_set, batch_size)
        else:
            raise ValueError(f'Emotion type "{emotions}" does not have data!')
