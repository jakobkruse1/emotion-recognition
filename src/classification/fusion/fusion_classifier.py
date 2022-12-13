""" This file contains a fusion emotion classifier fusing image, plant and
watch probabilities """

from typing import Dict

import numpy as np
import tensorflow as tf

from src.classification.emotion_classifier import EmotionClassifier
from src.data.data_reader import Set
from src.utils import logging, training_loop


class FusionClassifier(EmotionClassifier):
    """
    Class that implements the fusion emotion classifier.
    This classifier performs early fusion from the following classifiers:
    image, plant and watch (they can be excluded or included).
    We take the seven emotion probabilities as features from the three
    classifiers and then classify this data. We do not take the real features
    that the classifiers extract as our input data.

    To generate the data required to use this classifier, run the script:
    src/evaluation/scripts/continuous_data_creation.py
    """

    def __init__(self, parameters: Dict = None):
        """
        Initialize the fusion emotion classifier

        :param parameters: Some configuration parameters for the classifier
        """
        super().__init__(
            name="fusion", data_type="fusion", parameters=parameters
        )
        tf.get_logger().setLevel("ERROR")
        self.model = None
        self.logger = logging.KerasLogger()
        self.callback = None
        self.optimizer = None
        self.loss = None
        self.metrics = None
        self.logger.log_start({"init_parameters": parameters})

    def initialize_model(self, parameters: Dict) -> None:
        """
        Initializes a new fusion model architecture
        """
        hidden_size = parameters.get("hidden_size", 1024)
        input_elements = parameters.get("input_elements", 21)
        input_tensor = tf.keras.layers.Input(
            shape=(input_elements,), dtype=tf.float32, name="inputs"
        )
        hidden = tf.keras.layers.Dense(
            hidden_size,
            activation="relu",
            kernel_regularizer=tf.keras.regularizers.L1L2(l1=0.001, l2=0.001),
        )(input_tensor)
        hidden = tf.keras.layers.Dense(
            hidden_size,
            activation="relu",
            kernel_regularizer=tf.keras.regularizers.L1L2(l1=0.001, l2=0.001),
        )(hidden)
        top = tf.keras.layers.Dense(
            7, activation="softmax", name="classifier"
        )(hidden)
        self.model = tf.keras.Model(input_tensor, top)

    def train(self, parameters: Dict = None, **kwargs) -> None:
        """
        Training method for Fusion model

        :param parameters: Parameter dictionary used for training
        :param kwargs: Additional kwargs parameters
        """
        parameters = self.init_parameters(parameters, **kwargs)
        self.logger.log_start({"train_parameters": parameters})
        epochs = parameters.get("epochs", 50)

        if not self.model:
            self.initialize_model(parameters)
        self.prepare_training(parameters)
        self.model.compile(
            optimizer=self.optimizer, loss=self.loss, metrics=self.metrics
        )

        train_data = self.data_reader.get_seven_emotion_data(
            Set.TRAIN,
            batch_size=parameters.get("batch_size", 64),
            parameters=parameters,
        )
        val_data = self.data_reader.get_seven_emotion_data(
            Set.VAL,
            batch_size=parameters.get("batch_size", 64),
            parameters=parameters,
        )

        history = self.model.fit(
            x=train_data,
            validation_data=val_data,
            epochs=epochs,
            callbacks=[self.callback],
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
        save_path = parameters.get("save_path", "models/fusion/fusion")
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
        save_path = parameters.get("save_path", "models/fusion/fusion")
        self.model.save(save_path, include_optimizer=False)
        self.logger.save_logs(save_path)

    def classify(self, parameters: Dict = None, **kwargs) -> np.array:
        """
        Classification method used to classify emotions from images

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

    def prepare_training(self, parameters: Dict) -> None:
        """
        Function that prepares the training by initializing optimizer,
        loss, metrics and callbacks for training.

        :param parameters: Training parameters
        """
        learning_rate = parameters.get("learning_rate", 0.001)
        patience = parameters.get("patience", 5)

        self.callback = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=patience, restore_best_weights=True
        )
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.metrics = [tf.metrics.CategoricalAccuracy()]
        self.loss = tf.keras.losses.CategoricalCrossentropy()


def _main():  # pragma: no cover
    classifier = FusionClassifier()
    parameters = {
        "epochs": 500,
        "batch_size": 64,
        "patience": 15,
        "learning_rate": 0.003,
        "hidden_size": 64,
        "input_elements": 21,
    }
    save_path = "models/fusion/fusion"
    training_loop(classifier, parameters, save_path)


if __name__ == "__main__":  # pragma: no cover
    _main()
