""" This file contains the MFCC-LSTM speech emotion classifier """

from typing import Dict

import numpy as np
import tensorflow as tf

from src.classification.speech.speech_emotion_classifier import (
    SpeechEmotionClassifier,
)
from src.data.data_reader import Set
from src.utils import logging, training_loop


class MFCCLSTMClassifier(SpeechEmotionClassifier):
    """
    Class that implements a speech emotion classifier that uses mfc features
    in a LSTM based classifier model.
    """

    def __init__(self, parameters: Dict = None):
        """
        Initialize the MFCC-LSTM emotion classifier

        :param parameters: Some configuration parameters for the classifier
        """
        super().__init__("mfcc_lstm", parameters)
        tf.get_logger().setLevel("ERROR")
        self.model = None
        self.logger = logging.KerasLogger()
        self.logger.log_start({"init_parameters": parameters})

    def initialize_model(self, parameters: Dict) -> None:
        """
        Initializes a new and pretrained version of the MFCC-LSTM model

        :param parameters: Parameters for initializing the model.
        """
        lstm_units = parameters.get("lstm_units", 256)
        dropout = parameters.get("dropout", 0.2)
        input_tensor = tf.keras.layers.Input(
            shape=(48000), dtype=tf.float32, name="raw"
        )
        mfcc = self.compute_mfccs(input_tensor)
        out = tf.keras.layers.LSTM(lstm_units)(mfcc)
        out = tf.keras.layers.Dropout(dropout)(out)
        out = tf.keras.layers.Dense(1024, activation="relu")(out)
        out = tf.keras.layers.Dense(512, activation="relu")(out)
        out = tf.keras.layers.Dense(7, activation="softmax")(out)
        self.model = tf.keras.Model(input_tensor, out)

    def train(self, parameters: Dict = None, **kwargs) -> None:
        """
        Training method for MFCC-LSTM model

        :param parameters: Parameter dictionary used for training
        :param kwargs: Additional kwargs parameters
        """
        parameters = self.init_parameters(parameters, **kwargs)
        self.logger.log_start({"train_parameters": parameters})
        epochs = parameters.get("epochs", 20)

        if not self.model:
            self.initialize_model(parameters)
        self.prepare_training(parameters)
        self.model.compile(
            optimizer=self.optimizer, loss=self.loss, metrics=self.metrics
        )
        self.prepare_data(parameters)

        history = self.model.fit(
            x=self.train_data,
            validation_data=self.val_data,
            epochs=epochs,
            callbacks=[self.callback],
            class_weight=self.class_weights,
        )
        self.logger.log_end({"history": history})
        self.is_trained = True

    def load(self, parameters: Dict = None, **kwargs) -> None:
        """
        Loading method that loads a previously trained model from disk.

        :param parameters: Parameters required for loading the model
        :param kwargs: Additional kwargs parameters
        """
        parameters = self.init_parameters(parameters, **kwargs)
        save_path = parameters.get("save_path", "models/speech/mfcc_lstm")
        self.model = tf.keras.models.load_model(save_path)

    def save(self, parameters: Dict = None, **kwargs) -> None:
        """
        Saving method that saves a previously trained model on disk.

        :param parameters: Parameters required for storing the model
        :param kwargs: Additional kwargs parameters
        """
        if not self.is_trained:
            raise RuntimeError(
                "Model needs to be trained in order to save it!"
            )
        parameters = self.init_parameters(parameters, **kwargs)
        save_path = parameters.get("save_path", "models/speech/mfcc_lstm")
        self.model.save(save_path, include_optimizer=False)
        self.logger.save_logs(save_path)

    def classify(self, parameters: Dict = None, **kwargs) -> np.array:
        """
        Classification method used to classify emotions from speech

        :param parameters: Parameter dictionary used for classification
        :param kwargs: Additional kwargs parameters
        :return: An array with predicted emotion indices
        """
        parameters = self.init_parameters(parameters, **kwargs)
        which_set = parameters.get("which_set", Set.TEST)
        batch_size = parameters.get("batch_size", 64)
        dataset = self.data_reader.get_emotion_data(
            self.emotions, which_set, batch_size, parameters
        )

        if not self.model:
            raise RuntimeError(
                "Please load or train the model before inference!"
            )
        results = self.model.predict(dataset)
        return np.argmax(results, axis=1)


def _main():  # pragma: no cover
    classifier = MFCCLSTMClassifier()
    parameters = {
        "epochs": 50,
        "patience": 10,
        "learning_rate": 0.0003,
        "lstm_units": 128,
        "dropout": 0.2,
        "weighted": True,
    }
    save_path = "models/speech/mfcc_lstm"
    training_loop(classifier, parameters, save_path)


if __name__ == "__main__":  # pragma: no cover
    _main()
