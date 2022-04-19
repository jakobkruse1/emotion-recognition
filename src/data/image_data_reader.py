"""This file implements the data reading functionality for image data."""

import os

import numpy as np
import tensorflow as tf

from src.data.data_reader import DataReader, Set


class ImageDataReader(DataReader):
    """
    Class that reads the CSV datasets from the data/train/text folder
    """

    def __init__(self, folder: str = "data/train/image"):
        """
        Initialization for the class
        """
        super().__init__("text", folder)
        self.folder_map = {
            Set.TRAIN: "train",
            Set.VAL: "val",
            Set.TEST: "test",
        }

    def get_seven_emotion_data(
        self, which_set: Set, batch_size: int = 64, **kwargs
    ) -> tf.data.Dataset:
        """
        Main data reading function which reads the images into a dataset

        :param which_set: Which dataset to use - train, val or test
        :param batch_size: The batch size for the resulting dataset
        :param kwargs: Additional parameters
        :return: The tensorflow Dataset instance
        """
        shuffle = kwargs.get(
            "shuffle", True if which_set == Set.TRAIN else False
        )
        dataset = tf.keras.utils.image_dataset_from_directory(
            os.path.join(self.folder, self.folder_map[which_set]),
            shuffle=shuffle,
            batch_size=batch_size,
            image_size=(48, 48),
            label_mode="categorical",
            color_mode="grayscale",
            class_names=[
                "angry",
                "surprise",
                "disgust",
                "happy",
                "fear",
                "sad",
                "neutral",
            ],
        )
        return dataset

    def get_three_emotion_data(
        self, which_set: Set, batch_size: int = 64, **kwargs
    ) -> tf.data.Dataset:
        """
        Main data reading function which reads the CSV file into a dataset
        and also converts the emotion labels to the three emotion space.

        :param which_set: Which dataset to use - train, val or test
        :param batch_size: The batch size for the resulting dataset
        :param kwargs: Additional arguments
        :return: The tensorflow Dataset instance
        """
        folder_path = os.path.join(self.folder, self.folder_map[which_set])
        image_count = sum([len(files) for r, d, files in os.walk(folder_path)])
        shuffle = kwargs.get(
            "shuffle", True if which_set == Set.TRAIN else False
        )
        list_ds = tf.data.Dataset.list_files(
            f"{folder_path}/*/*", shuffle=False
        )
        if shuffle:
            list_ds = list_ds.shuffle(
                image_count, reshuffle_each_iteration=False
            )

        def get_label(file_path):
            # Convert the path to a list of path components
            label = tf.strings.split(file_path, os.path.sep)[-2]
            new_label = self.convert_to_three_emotions(np.array([label]))[0]
            return new_label

        def decode_img(img):
            # Convert the compressed string to a 3D uint8 tensor
            img = tf.io.decode_jpeg(img, channels=1)
            # Resize the image to the desired size
            return tf.image.resize(img, [48, 48])

        def process_path(file_path):
            label = get_label(file_path)
            label = tf.one_hot(tf.cast(label, tf.int32), 3)
            # Load the raw data from the file as a string
            img = tf.io.read_file(file_path)
            img = decode_img(img)
            return img, label

        dataset = list_ds.map(
            process_path, num_parallel_calls=tf.data.AUTOTUNE
        )
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

        return dataset

    def get_labels(self, which_set: Set = Set.TRAIN) -> np.ndarray:
        """
        Get the labels for the text dataset that is specified in an array

        :param which_set: Train, val or test set
        :return: The labels in an array of shape (num_samples,)
        """
        dataset = self.get_seven_emotion_data(which_set, shuffle=False)
        all_labels = np.empty((0,))
        for images, labels in dataset:
            all_labels = np.concatenate(
                [all_labels, np.argmax(labels.numpy(), axis=1)], axis=0
            )

        return all_labels
