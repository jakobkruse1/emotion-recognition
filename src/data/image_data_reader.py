"""This file implements the data reading functionality for image data."""

import os
from typing import Dict, Tuple

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

from src.data.data_reader import DataReader, Set


class ImageDataReader(DataReader):
    """
    Class that reads the CSV datasets from the data/train/text folder
    """

    def __init__(self, folder: str = None):
        """
        Initialization for the class
        """
        super().__init__("image", folder or "data/train/image")
        self.folder_map = {
            Set.TRAIN: "train",
            Set.VAL: "val",
            Set.TEST: "test",
        }

    def get_seven_emotion_data(
        self, which_set: Set, batch_size: int = 64, parameters: Dict = None
    ) -> tf.data.Dataset:
        """
        Main data reading function which reads the images into a dataset

        :param which_set: Which dataset to use - train, val or test
        :param batch_size: The batch size for the resulting dataset
        :param parameters: Additional parameters
        :return: The tensorflow Dataset instance
        """
        parameters = parameters or {}
        shuffle = parameters.get(
            "shuffle", True if which_set == Set.TRAIN else False
        )
        augment = parameters.get(
            "augment", True if which_set == Set.TRAIN else False
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
        dataset = self.add_augmentations(dataset, augment)
        return dataset

    def get_three_emotion_data(
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
        shuffle = parameters.get(
            "shuffle", True if which_set == Set.TRAIN else False
        )
        augment = parameters.get(
            "augment", True if which_set == Set.TRAIN else False
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
        dataset = dataset.map(
            lambda x, y: tf.numpy_function(
                func=self.map_emotions,
                inp=[x, y],
                Tout=(tf.float32, tf.float32),
            )
        )
        dataset = self.add_augmentations(dataset, augment)
        return dataset

    @staticmethod
    def map_emotions(data, labels):
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

    def add_augmentations(
        self, dataset: tf.data.Dataset, use_augmentations: bool = True
    ):
        """
        Function that adds augmentation to the dataset.
        This helps reduce overfitting of the model.
        :param dataset: The dataset containing images
        :param use_augmentations: Boolean flag to enable augmentation
        :return: The dataset with augmented images
        """
        if not use_augmentations:
            return dataset

        counter = tf.data.experimental.Counter()
        dataset = tf.data.Dataset.zip((dataset, (counter, counter)))
        augmented_dataset = dataset.map(self._augment)

        return augmented_dataset

    @staticmethod
    @tf.function
    def _augment(
        data, seed
    ) -> Tuple[tf.Tensor, tf.Tensor]:  # pragma: no cover
        """
        Augmentation function that rotates, brightens, and flips images.

        :param data: The data to perform augmentation on.
            Contains tuples of images and labels.
        :param seed: Random seed to use for augmentation
        :return: images and labels that are augmented
        """
        image, label = data
        new_seed = tf.random.experimental.stateless_split(seed, num=1)[0, :]
        image = tf.image.stateless_random_brightness(
            image, max_delta=0.3, seed=new_seed
        )
        image = tf.clip_by_value(image, 0.0, 255.0)
        image = tf.image.stateless_random_flip_left_right(image, seed=new_seed)
        num_samples = int(tf.shape(image)[0])
        degrees = tf.random.stateless_uniform(
            shape=(num_samples,), seed=seed, minval=-45, maxval=45
        )
        degrees = degrees * 0.017453292519943295
        image = tfa.image.rotate(image, degrees)
        return image, label
