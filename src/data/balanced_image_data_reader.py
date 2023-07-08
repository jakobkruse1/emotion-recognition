"""This file implements an image data reader which balances data."""

import os
from typing import Dict

import numpy as np
import tensorflow as tf

from src.data.image_data_reader import ImageDataReader, Set
from src.utils import reader_main


class BalancedImageDataReader(ImageDataReader):
    """
    Class that reads images from folders in a balanced way.
    This means that of all classes, there should be an approximately equal
    amount of images from that class. This means that some images from
    underrepresented classes might appear twice and some images from
    overrepresented classes might not appear at all.
    Note: Has higher memory requirements than other Data Readers.
    """

    def __init__(self, folder: str = None):
        """
        Initialization for the class

        :param folder: folder that contains the data.
        """
        super().__init__(
            "balanced_image", folder or os.path.join("data", "train", "image")
        )

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
        if parameters.get("balanced", False):
            raise NotImplementedError(
                "Balanced dataset not existing for three emotions!"
            )
        else:
            return self._get_unbalanced_three_emotion_data(
                which_set, batch_size, parameters
            )

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
        Main data reading function which reads the images into a dataset.
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
        augment = parameters.get(
            "augment", True if which_set == Set.TRAIN else False
        )
        class_data = [np.empty((0, 48, 48, 1)) for _ in range(7)]
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
        for images, labels in self._get_unbalanced_seven_emotion_data(
            which_set, 1024, {"shuffle": False, "augment": False}
        ):
            image_class = np.argmax(labels.numpy(), axis=1)
            images = images.numpy()
            for index in range(7):
                class_data[index] = np.concatenate(
                    [class_data[index], images[image_class == index, :, :, :]],
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

        resampled_ds = self.add_augmentations(resampled_ds, augment)
        return resampled_ds

    def _get_unbalanced_seven_emotion_data(
        self, which_set: Set, batch_size: int = 64, parameters: Dict = None
    ) -> tf.data.Dataset:
        """
        Main data reading function which reads the images into a dataset.
        No additional balancing is performed here.

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

    def _get_unbalanced_three_emotion_data(
        self, which_set: Set, batch_size: int = 64, parameters: Dict = None
    ) -> tf.data.Dataset:
        """
        Main data reading function which reads the CSV file into a dataset
        and also converts the emotion labels to the three emotion space.
        No additional balancing is performed here.

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
        dataset = self._get_unbalanced_seven_emotion_data(
            which_set, parameters=parameters
        )
        all_labels = np.empty((0,))
        for images, labels in dataset:
            all_labels = np.concatenate(
                [all_labels, np.argmax(labels.numpy(), axis=1)], axis=0
            )

        return all_labels


def _main():  # pragma: no cover
    reader = BalancedImageDataReader()
    reader_main(reader, {"balanced": True})


if __name__ == "__main__":  # pragma: no cover
    _main()
