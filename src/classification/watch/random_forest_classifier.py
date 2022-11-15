""" This file defines a Random Forest classifier for the watch data. """

import copy
import os
import pickle
from typing import Dict

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from src.classification.watch.watch_emotion_classifier import (
    WatchEmotionClassifier,
)
from src.data.data_reader import Set
from src.utils import cv_training_loop, logging


class WatchRandomForestClassifier(WatchEmotionClassifier):
    """
    Random Forest classifier for watch data.
    """

    def __init__(self, parameters: Dict = None):
        """
        Initialize the Random Forest watch classifier.

        :param parameters: Some configuration parameters for the classifier
        """
        super().__init__("random_forest", parameters)
        self.model = None
        self.logger = logging.StandardLogger()
        self.logger.log_start({"init_parameters": parameters})

    def train(self, parameters: Dict = None, **kwargs) -> None:
        """
        Training method for Random Forest classifier

        :param parameters: Parameter dictionary used for training
        :param kwargs: Additional kwargs parameters
        """
        parameters = self.init_parameters(parameters, **kwargs)
        self.logger.log_start({"train_parameters": parameters})
        n_estimators = parameters.get("n_estimators", 83)
        min_samples_split = parameters.get("min_samples_split", 4)
        max_depth = parameters.get("max_depth", 49)
        window = parameters.get("window", 20)

        self.prepare_training(parameters)
        self.prepare_data(parameters)

        X = np.empty((0, window, 5))
        y = np.empty((0,))
        for batch, labels in self.train_data:
            X = np.concatenate([X, batch], axis=0)
            y = np.concatenate([y, np.argmax(labels, 1)], axis=0)

        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
        )
        X = np.reshape(X, (X.shape[0], -1))
        self.model.fit(X, y)

        self.logger.log_end({"history": None})
        self.is_trained = True

    def load(self, parameters: Dict = None, **kwargs) -> None:
        """
        Loading method that loads a previously trained model from disk.

        :param parameters: Parameters required for loading the model
        :param kwargs: Additional kwargs parameters
        """
        parameters = self.init_parameters(parameters, **kwargs)
        save_path = parameters.get("save_path", "models/watch/random_forest")
        model_path = os.path.join(save_path, "model.pkl")
        with open(model_path, "rb") as file:
            self.model = pickle.load(file)

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
        save_path = parameters.get("save_path", "models/watch/random_forest")
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        model_path = os.path.join(save_path, "model.pkl")
        with open(model_path, "wb") as file:
            pickle.dump(self.model, file)
        self.logger.save_logs(save_path)

    def classify(self, parameters: Dict = None, **kwargs) -> np.array:
        """
        Classification method used to classify emotions from watch data

        :param parameters: Parameter dictionary used for classification
        :param kwargs: Additional kwargs parameters
        :return: An array with predicted emotion indices
        """
        parameters = self.init_parameters(parameters, **kwargs)
        which_set = parameters.get("which_set", Set.TEST)
        batch_size = parameters.get("batch_size", 64)
        eval_parameters = copy.deepcopy(parameters)
        eval_parameters["balanced"] = False
        dataset = self.data_reader.get_emotion_data(
            self.emotions, which_set, batch_size, eval_parameters
        )

        if not self.model:
            raise RuntimeError(
                "Please load or train the model before inference!"
            )

        window = parameters.get("window", 20)
        X = np.empty((0, window, 5))
        for batch, labels in dataset:
            X = np.concatenate([X, batch], axis=0)
        X = np.reshape(X, (X.shape[0], -1))
        results = self.model.predict_proba(X)
        return np.argmax(results, axis=1)


def _main():  # pragma: no cover
    classifier = WatchRandomForestClassifier()
    parameters = {
        "label_mode": "both",
        "batch_size": 64,
        "window": 20,
        "hop": 2,
        "balanced": True,
        "max_depth": 30,
        "n_estimators": 10,
        "min_samples_split": 4,
    }
    save_path = "models/watch/random_forest"
    cv_training_loop(classifier, parameters, save_path)


if __name__ == "__main__":  # pragma: no cover
    _main()
