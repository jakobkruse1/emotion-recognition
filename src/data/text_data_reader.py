"""This file implements the data reading functionality for text data."""

import tensorflow as tf

from src.data.data_reader import DataReader, Set


class TextDataReader(DataReader):
    """
    Class that reads the CSV datasets from the data/train/text folder
    """

    def __init__(self):
        """
        Initialization for the class
        """
        super().__init__("text", "data/train/text")
        self.file_map = {
            Set.TRAIN: "final_train.csv",
            Set.VAL: "final_val.csv",
            Set.TEST: "final_test.csv",
        }

    def get_data(
        self, which_set: Set, batch_size: int = 64
    ) -> tf.data.Dataset:
        """
        Main data reading function which reads the CSV file into a dataset

        :param which_set: Which dataset to use - train, val or test
        :param batch_size: The batch size for the resulting dataset
        :return: The tensorflow Dataset instance
        """
        csv_file_path = self.file_map[which_set]
        dataset = tf.data.experimental.CsvDataset(
            csv_file_path, [tf.string, tf.int32], field_delim="\t"
        )
        dataset = (
            dataset.shuffle(1024)
            .cache()
            .batch(batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )
        return dataset

    def get_three_emotion_data(self, which_set: Set) -> tf.data.Dataset:
        """
        Main data reading function which reads the CSV file into a dataset
        and also converts the emotion labels to the three emotion space.

        :param which_set: Which dataset to use - train, val or test
        :return: The tensorflow Dataset instance
        """
        pass
