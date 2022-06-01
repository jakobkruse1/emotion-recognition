"""This file implements the data reading functionality for speech data."""
import os
from typing import Dict, Tuple

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from src.data.data_reader import DataReader, Set

CLASS_NAMES = [
    "angry",
    "surprise",
    "disgust",
    "happy",
    "fear",
    "sad",
    "neutral",
]

CREMA_LABELS = {0: 6, 1: 3, 2: 5, 3: 0, 4: 4, 5: 2}


class SpeechDataReader(DataReader):
    """
    Class that reads the speech datasets
    """

    def __init__(self, name: str = "speech", folder: str = None):
        """
        Initialization for the class
        """
        super().__init__(name, folder or "data/train/speech")
        self.folder_map = {
            Set.TRAIN: "train",
            Set.VAL: "val",
            Set.TEST: "test",
        }

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
        shuffle = parameters.get(
            "shuffle", True if which_set == Set.TRAIN else False
        )
        max_elements = parameters.get("max_elements", None)
        folder = self.folder_map[which_set]
        crema_d, cd_info = tfds.load(
            "crema_d",
            split="validation" if folder == "val" else folder,
            shuffle_files=False,
            with_info=True,
            as_supervised=True,
        )
        crema_d = crema_d.map(
            lambda x, y: tf.numpy_function(
                func=self.process_crema,
                inp=[x, y],
                Tout=[tf.float32, tf.float32],
            )
        )
        num_crema = cd_info.splits[
            "validation" if folder == "val" else folder
        ].num_examples
        if shuffle:
            crema_d.shuffle(1000)
        data_dir = os.path.join(self.folder, folder)
        filenames = tf.io.gfile.glob(str(data_dir) + "/*/*.wav")
        filenames.sort()
        num_train = len(filenames)
        if shuffle:
            filenames = tf.random.shuffle(filenames)
        files_ds = tf.data.Dataset.from_tensor_slices(filenames)
        wave_ds = files_ds.map(
            lambda p: tf.numpy_function(
                func=self.get_waveform_and_label,
                inp=[p],
                Tout=[tf.float32, tf.float32],
            )
        )
        if not shuffle:
            dataset = wave_ds.concatenate(crema_d)
        else:
            dataset = tf.data.Dataset.sample_from_datasets(
                [crema_d, wave_ds],
                weights=[
                    num_crema / (num_train + num_crema),
                    num_train / (num_train + num_crema),
                ],
            )
        if max_elements:
            dataset = dataset.take(max_elements)
        dataset = dataset.batch(batch_size)
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
        """
        new_labels = DataReader.convert_to_three_emotions_onehot(
            labels
        ).astype(np.float32)
        return data, new_labels

    def get_labels(self, which_set: Set = Set.TRAIN) -> np.ndarray:
        """
        Get the labels for the text dataset that is specified in an array

        :param which_set: Train, val or test set
        :return: The labels in an array of shape (num_samples,)
        """
        dataset = self.get_seven_emotion_data(
            which_set, parameters={"shuffle": False}
        )
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
        y = tf.keras.utils.to_categorical(label, num_classes=7)
        return audio, y

    @staticmethod
    def process_crema(x: np.ndarray, y: int) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Preprocessing function for the crema dataset read from
        tensorflow_datasets package.

        :param x: The audio data
        :param y: The label data
        :return: Processed audio and label data
        """
        audio = tf.cast(x, tf.float32) / 32768
        audio = tf.keras.preprocessing.sequence.pad_sequences(
            [audio],
            maxlen=48000,
            dtype="float32",
            padding="pre",
            truncating="post",
            value=0,
        )[0]
        y = CREMA_LABELS[y]
        y = tf.keras.utils.to_categorical(y, num_classes=7)
        return audio, y


if __name__ == "__main__":  # pragma: no cover
    dr = SpeechDataReader()
    ds = dr.get_seven_emotion_data(Set.TRAIN, batch_size=64)

    for data, labels in ds:
        print(f"{data.shape} | {labels.shape}")
