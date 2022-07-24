import os
from typing import Dict

import numpy as np
import tensorflow as tf

from src.classification.speech import SpeechEmotionClassifier


class DummySpeechClassifier(SpeechEmotionClassifier):
    def train(self, parameters: Dict = None, **kwargs) -> None:
        self.prepare_training(parameters)
        self.prepare_data(parameters)

    def load(self, parameters: Dict = None, **kwargs) -> None:
        pass

    def save(self, parameters: Dict = None, **kwargs) -> None:
        pass

    def classify(self, parameters: Dict = None, **kwargs) -> np.array:
        pass


def test_training_preparation():
    classifier = DummySpeechClassifier()
    classifier.data_reader.folder = "tests/test_data/speech"
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
    classifier = DummySpeechClassifier()
    assert classifier.train_data is None
    assert classifier.val_data is None
    assert classifier.class_weights is None
    classifier.data_reader.folder = "tests/test_data/speech"
    parameters = {"batch_size": 5, "weighted": False, "dataset": "meld"}
    classifier.train(parameters)

    assert isinstance(classifier.train_data, tf.data.Dataset)
    assert isinstance(classifier.val_data, tf.data.Dataset)
    assert classifier.class_weights is None

    parameters = {"batch_size": 5, "weighted": True, "dataset": "meld"}
    classifier.train(parameters)
    assert classifier.class_weights is not None
    for i in range(7):
        assert classifier.class_weights[i] == 1
    try:
        import shutil

        shutil.copyfile(
            "tests/test_data/speech/train/angry/03-01-05-01-01-01-02.wav",
            "tests/test_data/speech/train/angry/03-01-05-01-01-01-02_copy.wav",
        )
        classifier.train(parameters)
        os.remove(
            "tests/test_data/speech/train/angry/03-01-05-01-01-01-02_copy.wav"
        )
    except BaseException as e:
        if os.path.exists(
            "tests/test_data/speech/train/angry/03-01-05-01-01-01-02.wav"
        ):
            os.remove(
                "tests/test_data/speech/train/angry/"
                "03-01-05-01-01-01-02_copy.wav"
            )
        raise e

    assert classifier.class_weights[0] == 8.0 / 14.0
    for i in range(1, 7):
        assert classifier.class_weights[i] == 8.0 / 7.0
