"""This file implements the data reading functionality for the speech data
from the comparison dataset."""
import os
import warnings
from typing import Dict, Tuple

import numpy as np
import tensorflow as tf

from src.data.data_reader import DataReader, Set
from src.utils import reader_main

CLASS_NAMES = [
    "angry",
    "surprise",
    "disgust",
    "happy",
    "fear",
    "sad",
    "neutral",
]


class ComparisonSpeechDataReader(DataReader):
    """
    Class that reads the comparison speech dataset
    """

    def __init__(self, name: str = "comparison_speech", folder: str = None):
        """
        Initialization for the class

        :param name: The name of the data reader, speech
        :param folder: Folder that contains the data.
        """
        super().__init__(name, folder or "data/comparison_dataset/audio")

    def get_seven_emotion_data(
        self, which_set: Set, batch_size: int = 64, parameters: Dict = None
    ) -> tf.data.Dataset:
        """
        Main data reading function which reads the audio files into a dataset

        :param which_set: Which dataset to use - train, val or test
        :param batch_size: The batch size for the resulting dataset
        :param parameters: Additional parameters
        :return: The tensorflow Dataset instance
        """
        parameters = parameters or {}

        filenames = tf.io.gfile.glob(self.folder + "/*/*.wav")
        filenames.sort()
        num_files = len(filenames)
        files_ds = tf.data.Dataset.from_tensor_slices(filenames)
        dataset = files_ds.map(
            lambda p: tf.numpy_function(
                func=self.get_waveform_and_label,
                inp=[p],
                Tout=[tf.float32, tf.float32],
            )
        )

        self.num_batch[which_set] = int(np.ceil(num_files / batch_size))

        dataset = dataset.batch(batch_size)
        dataset = dataset.map(map_func=self.set_tensor_shapes)
        return dataset

    def get_three_emotion_data(
        self, which_set: Set, batch_size: int = 64, parameters: Dict = None
    ) -> tf.data.Dataset:
        """
        Main data reading function which reads the audio data from disk.

        :param which_set: Which dataset to use - train, val or test
        :param batch_size: The batch size for the resulting dataset
        :param parameters: Additional arguments
        :return: The tensorflow Dataset instance
        """
        parameters = parameters or {}
        seven_dataset = self.get_seven_emotion_data(
            which_set, batch_size, parameters
        )
        three_dataset = seven_dataset.map(
            lambda x, y: tf.numpy_function(
                func=self.map_emotions,
                inp=[x, y],
                Tout=(tf.float32, tf.float32),
            )
        )

        return three_dataset

    @staticmethod
    def map_emotions(data: np.ndarray, labels: np.ndarray):
        """
        Conversion function that is applied when three emotion labels are
        required.

        :param data: The emotional data.
        :param labels: The labels that need to be converted to three emotions.
        """
        new_labels = DataReader.convert_to_three_emotions_onehot(
            labels
        ).astype(np.float32)
        return data, new_labels

    def get_labels(
        self, which_set: Set = Set.TRAIN, parameters: Dict = None
    ) -> np.ndarray:
        """
        Get the labels for the text dataset that is specified in an array

        :param which_set: Train, val or test set
        :param parameters: Parameter dictionary
        :return: The labels in an array of shape (num_samples,)
        """
        parameters = parameters or {}
        dataset = self.get_seven_emotion_data(which_set, parameters=parameters)
        all_labels = np.empty((0,))
        for images, labels in dataset:
            all_labels = np.concatenate(
                [all_labels, np.argmax(labels.numpy(), axis=1)], axis=0
            )

        return all_labels

    @staticmethod
    def get_waveform_and_label(
        file_path: bytes,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Preprocessing function for the audio files that are read from the data
        folder. Files are read, decoded and padded or truncated.

        :param file_path: The path of one audio file to read.
        :return: Audio tensor and label tensor in a tuple
        """
        audio_binary = tf.io.read_file(file_path)
        audio, _ = tf.audio.decode_wav(
            contents=audio_binary, desired_channels=1
        )
        if audio.shape[0] > 48000:
            warnings.warn(f"Truncating audio file of size {audio.shape}")
        audio = tf.keras.preprocessing.sequence.pad_sequences(
            [audio],
            maxlen=48000,
            dtype="float32",
            padding="pre",
            truncating="post",
            value=0,
        )[0]
        audio = tf.squeeze(audio, axis=-1, name="audio")
        emotion = file_path.decode("utf-8").split(os.path.sep)[-2]
        label = CLASS_NAMES.index(emotion)
        y = tf.convert_to_tensor(
            tf.keras.utils.to_categorical(label, num_classes=7)
        )
        return audio, y

    @staticmethod
    def set_tensor_shapes(
        x: tf.Tensor, y: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Function that sets the tensor shapes in the dataset manually.
        This fixes an issue where using Dataset.map and numpy_function causes
        the tensor shape to be unknown.
        See the issue here:
        https://github.com/tensorflow/tensorflow/issues/47032

        :param x: The speech tensor
        :param y: The labels tensor
        :return: Tuple with speech and labels tensor
        """
        x.set_shape([None, 48000])
        y.set_shape([None, 7])
        return x, y


def _main():  # pragma: no cover
    reader = ComparisonSpeechDataReader()
    reader_main(reader, {})


if __name__ == "__main__":  # pragma: no cover
    _main()
