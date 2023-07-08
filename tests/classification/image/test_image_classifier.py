import os
from typing import Dict

import numpy as np
import pytest
import tensorflow as tf

from src.classification import EmotionClassifier
from src.classification.image import ImageEmotionClassifier
from src.data.balanced_image_data_reader import BalancedImageDataReader
from src.data.image_data_reader import ImageDataReader, Set


class DummyImageClassifier(ImageEmotionClassifier):
    def train(self, parameters: Dict = None, **kwargs) -> None:
        self.prepare_training(parameters, **kwargs)
        self.prepare_data(parameters, **kwargs)

    def load(self, parameters: Dict = None, **kwargs) -> None:
        pass

    def save(self, parameters: Dict = None, **kwargs) -> None:
        pass

    def classify(self, parameters: Dict = None, **kwargs) -> np.array:
        pass


def test_training_preparation():
    classifier = DummyImageClassifier()
    classifier.data_reader.folder = os.path.join("tests", "test_data", "image")
    assert classifier.callback is None
    assert classifier.optimizer is None
    assert classifier.loss is None
    assert classifier.metrics is None

    parameters = {"patience": 11, "learning_rate": 77}

    classifier.train(parameters)
    assert isinstance(classifier.callback, tf.keras.callbacks.Callback)
    assert classifier.callback.patience == 11
    assert isinstance(classifier.optimizer, tf.keras.optimizers.Adam)
    assert classifier.optimizer.learning_rate == 77
    assert isinstance(classifier.loss, tf.keras.losses.CategoricalCrossentropy)
    assert isinstance(classifier.metrics, list)
    assert len(classifier.metrics) == 1


def test_data_preparation():
    classifier = DummyImageClassifier()
    assert classifier.train_data is None
    assert classifier.val_data is None
    assert classifier.class_weights is None
    classifier.data_reader = ImageDataReader(
        folder=os.path.join("tests", "test_data", "image")
    )
    parameters = {"which_set": Set.TRAIN, "batch_size": 5, "weighted": False}
    classifier.train(parameters)

    assert isinstance(classifier.train_data, tf.data.Dataset)
    assert isinstance(classifier.val_data, tf.data.Dataset)
    assert classifier.class_weights is None

    parameters = {"which_set": Set.TRAIN, "batch_size": 5, "weighted": True}
    classifier.train(parameters)
    assert classifier.class_weights is not None
    for i in range(7):
        assert classifier.class_weights[i] == 1
    try:
        import shutil

        shutil.copyfile(
            os.path.join(
                "tests",
                "test_data",
                "image",
                "train",
                "angry",
                "fer_35854.jpeg",
            ),
            os.path.join(
                "tests",
                "test_data",
                "image",
                "train",
                "angry",
                "fer_35854_copy.jpeg",
            ),
        )
        classifier.train(parameters)
        os.remove(
            os.path.join(
                "tests",
                "test_data",
                "image",
                "train",
                "angry",
                "fer_35854_copy.jpeg",
            )
        )
    except BaseException as e:
        if os.path.exists(
            os.path.join(
                "tests",
                "test_data",
                "image",
                "train",
                "angry",
                "fer_35854_copy.jpeg",
            )
        ):
            os.remove(
                os.path.join(
                    "tests",
                    "test_data",
                    "image",
                    "train",
                    "angry",
                    "fer_35854_copy.jpeg",
                )
            )
        raise e

    assert classifier.class_weights[0] == 8.0 / 14.0
    for i in range(1, 7):
        assert classifier.class_weights[i] == 8.0 / 7.0


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_balanced_switch():
    classifier = DummyImageClassifier()
    assert classifier.train_data is None
    assert classifier.val_data is None
    assert classifier.class_weights is None
    classifier.data_reader = ImageDataReader(
        folder=os.path.join("tests", "test_data", "image")
    )
    parameters = {"which_set": Set.TRAIN, "batch_size": 5, "balanced": False}
    classifier.train(parameters)
    assert isinstance(classifier.data_reader, ImageDataReader)

    parameters = {"which_set": Set.TRAIN, "batch_size": 5, "balanced": True}
    classifier.train(parameters)
    assert isinstance(classifier.data_reader, BalancedImageDataReader)

    with pytest.warns(UserWarning):
        parameters = {"balanced": True, "weighted": True}
        classifier.train(parameters)


def test_init_parameters():
    parameters = {"test": 1, "var": "hello there", "override": False}

    new_parameters = EmotionClassifier.init_parameters(
        parameters, new_var=44, override=True
    )

    assert new_parameters["test"] == 1
    assert new_parameters["var"] == "hello there"
    assert new_parameters["override"]
    assert new_parameters["new_var"] == 44

    def proxy_function(parameters, **kwargs):
        return EmotionClassifier.init_parameters(parameters, **kwargs)

    assert new_parameters == proxy_function(
        parameters, new_var=44, override=True
    )
