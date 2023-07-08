"""This file implements classwise data reading for speech data."""
import os
from typing import Dict, Generator, Tuple

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


class ClasswiseSpeechDataReader(DataReader):
    """
    Class that reads the speech datasets per class.
    This means that the data extraction methods return one array per class.
    This is required for HMM and GMM classifiers which need all data for one
    class at the same time and do not support batching like NNs.
    """

    def __init__(self, name: str = "classwise_speech", folder: str = None):
        """
        Initialization for the class

        :param name: name of the data reader.
        :param folder: folder that contains the data.
        """
        super().__init__(
            name, folder or os.path.join("data", "train", "speech")
        )
        self.folder_map = {
            Set.TRAIN: "train",
            Set.VAL: "val",
            Set.TEST: "test",
        }

    def get_seven_emotion_data(
        self, which_set: Set, batch_size: int = 64, parameters: Dict = None
    ) -> Generator[Tuple[np.ndarray, str], None, None]:
        """
        Main data reading function which reads the audio files
        and then returns them one class at a time.

        :param which_set: Which dataset to use - train, val or test
        :param batch_size: The batch size for the resulting dataset
        :param parameters: Additional parameters
        :return: Generator that yields (array, class name)
        """
        parameters = parameters or {}
        shuffle = parameters.get(
            "shuffle", True if which_set == Set.TRAIN else False
        )
        dataset = parameters.get("dataset", "all")
        use_meld = dataset == "all" or dataset == "meld"
        use_crema = dataset == "all" or dataset == "crema"
        assert use_crema or use_meld
        folder = self.folder_map[which_set]
        if use_crema:
            crema_d, cd_info = tfds.load(
                "crema_d",
                split="validation" if folder == "val" else folder,
                shuffle_files=False,
                with_info=True,
                as_supervised=True,
                download=parameters.get("download", True),
            )
        data_dir = os.path.join(self.folder, folder)
        for emotion_class in CLASS_NAMES:
            if not use_meld:
                all_samples = self.get_crema_samples(crema_d, emotion_class)
            elif not use_crema:
                all_samples = self.get_file_samples(emotion_class, data_dir)
            else:
                all_samples = np.concatenate(
                    [
                        self.get_crema_samples(crema_d, emotion_class),
                        self.get_file_samples(emotion_class, data_dir),
                    ]
                )
            if shuffle:
                np.random.shuffle(all_samples)
            yield (all_samples, emotion_class)

    def get_crema_samples(
        self, crema_d: tf.data.Dataset, class_name: str
    ) -> np.ndarray:
        """
        Gets the samples from a specified class from the crema dataset

        :param crema_d: The entire crema dataset instance
        :param class_name: The class to extract from crema_d
        :return: A numpy array with the extracted data
        """
        crema_ds = crema_d.map(
            lambda x, y: tf.numpy_function(
                func=self.process_crema,
                inp=[x, y],
                Tout=[tf.float32, tf.float32],
            )
        )
        class_id = CLASS_NAMES.index(class_name)
        crema_ds = crema_ds.batch(100)
        all_data = np.empty((0, 48000))
        for data, labels in crema_ds:
            labels = np.argmax(labels.numpy(), axis=1)
            class_data = data.numpy()[labels == class_id, :]
            all_data = np.concatenate([all_data, class_data])
        return all_data

    def get_file_samples(
        self, emotion_class: str, data_dir: str
    ) -> np.ndarray:
        """
        Extract the data from a specific class from disk

        :param emotion_class: The class to load from disk
        :param data_dir: The directory on disk that contains the data
        :return: Numpy array with the data
        """
        filenames = tf.io.gfile.glob(
            f"{str(data_dir)}{os.sep}{emotion_class}{os.sep}*.wav"
        )
        files_ds = tf.data.Dataset.from_tensor_slices(filenames)
        wave_ds = files_ds.map(
            lambda p: tf.numpy_function(
                func=self.get_waveform_and_label,
                inp=[p],
                Tout=[tf.float32, tf.float32],
            )
        )
        wave_ds = wave_ds.batch(1000)
        all_data = np.empty((0, 48000))
        for data, _ in wave_ds:
            all_data = np.concatenate([all_data, data.numpy()], axis=0)

        return all_data

    def get_three_emotion_data(
        self, which_set: Set, batch_size: int = 64, parameters: Dict = None
    ) -> Generator[Tuple[np.ndarray, str], None, None]:
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
        neg_classes = ["angry", "disgust", "fear", "sad"]
        neu_classes = ["neutral"]
        pos_classes = ["surprise", "happy"]
        neg_data = np.empty((0, 48000))
        neu_data = np.empty((0, 48000))
        pos_data = np.empty((0, 48000))
        for data, class_name in seven_dataset:
            if class_name in neg_classes:
                neg_data = np.concatenate([neg_data, data])
            elif class_name in neu_classes:
                neu_data = np.concatenate([neu_data, data])
            elif class_name in pos_classes:
                pos_data = np.concatenate([pos_data, data])
        to_return = [
            (neg_data, "negative"),
            (neu_data, "neutral"),
            (pos_data, "positive"),
        ]
        for three_tuple in to_return:
            yield three_tuple

    @staticmethod
    def map_emotions(data: np.ndarray, labels: np.ndarray):
        """
        Conversion function that is applied when three emotion labels are
        required.

        :param data: The emotions data.
        :param labels: The labels that are to be converted to three emotions.
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
        parameters.update({"shuffle": False})
        dataset = self.get_seven_emotion_data(which_set, parameters=parameters)
        all_labels = np.empty((0,))
        for images, class_name in dataset:
            labels = np.ones((images.shape[0],)) * CLASS_NAMES.index(
                class_name
            )
            all_labels = np.concatenate([all_labels, labels], axis=0)

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
        y = tf.convert_to_tensor(
            tf.keras.utils.to_categorical(label, num_classes=7)
        )
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
        y = tf.convert_to_tensor(
            tf.keras.utils.to_categorical(y, num_classes=7)
        )
        return audio, y


def _main():  # pragma: no cover
    dr = ClasswiseSpeechDataReader()
    ds = dr.get_seven_emotion_data(Set.VAL, batch_size=-1)

    for data, class_name in ds:
        print(f"{class_name}: {data.shape}")

    ds = dr.get_three_emotion_data(Set.VAL, batch_size=-1)

    for data, class_name in ds:
        print(f"{class_name}: {data.shape}")


if __name__ == "__main__":  # pragma: no cover
    _main()
