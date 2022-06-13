""" This file contains the MFCC-LSTM speech emotion classifier """
import os
import pickle
from typing import Dict

import numpy as np
from sklearn import preprocessing
from sklearn.mixture import GaussianMixture

from src.classification.speech.speech_emotion_classifier import (
    SpeechEmotionClassifier,
)
from src.data.classwise_speech_data_reader import ClasswiseSpeechDataReader
from src.data.data_reader import Set

CLASS_NAMES = [
    "angry",
    "surprise",
    "disgust",
    "happy",
    "fear",
    "sad",
    "neutral",
]


class GMMClassifier(SpeechEmotionClassifier):
    """
    Class that implements a speech emotion classifier that uses
    a Gaussian Mixture Model for classification
    """

    def __init__(self, parameters: Dict = None):
        """
        Initialize the GMMemotion classifier

        :param name: The name for the classifier
        :param parameters: Some configuration parameters for the classifier
        """
        super().__init__("gmm", parameters)
        self.models = {}
        self.data_reader = ClasswiseSpeechDataReader(
            folder=self.data_reader.folder
        )
        self.scaler = {}

    def train(self, parameters: Dict = None, **kwargs) -> None:
        """
        Training method for HMM model

        :param parameters: Parameter dictionary used for training
        :param kwargs: Additional kwargs parameters
        """
        parameters = self.init_parameters(parameters, **kwargs)
        which_set = parameters.get("which_set", Set.TRAIN)
        self.train_data = self.data_reader.get_emotion_data(
            self.emotions, which_set, -1, parameters
        )
        n_components = parameters.get("n_components", 4)

        for data, class_name in self.train_data:
            features = np.empty((0, parameters.get("mfcc_num", 40) + 4))
            for index, sample in enumerate(data):
                sample_features = np.reshape(
                    self.get_mixed_features(sample, parameters), (1, -1)
                )
                features = np.concatenate([features, sample_features], axis=0)
            self.scaler[class_name] = preprocessing.StandardScaler().fit(
                features
            )
            tr_fea = self.scaler[class_name].transform(features)
            model = GaussianMixture(n_components)
            model.fit(tr_fea)
            self.models[class_name] = model

        self.is_trained = True

    def load(self, parameters: Dict = None, **kwargs) -> None:
        """
        Loading method that loads a previously trained model from disk.

        :param parameters: Parameters required for loading the model
        :param kwargs: Additional kwargs parameters
        """
        parameters = self.init_parameters(parameters, **kwargs)
        save_path = parameters.get("save_path", "models/speech/gmm")
        for class_name in CLASS_NAMES:
            model_path = os.path.join(save_path, f"{class_name}.pkl")
            with open(model_path, "rb") as file:
                model = pickle.load(file)
            self.models[class_name] = model
            scaler_path = os.path.join(save_path, f"{class_name}_scaler.pkl")
            with open(scaler_path, "rb") as file:
                scaler = pickle.load(file)
            self.scaler[class_name] = scaler
        self.is_trained = True

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
        save_path = parameters.get("save_path", "models/speech/gmm")
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        for name, model in self.models.items():
            model_path = os.path.join(save_path, f"{name}.pkl")
            with open(model_path, "wb") as file:
                pickle.dump(model, file)
        for name, scaler in self.scaler.items():
            scaler_path = os.path.join(save_path, f"{name}_scaler.pkl")
            with open(scaler_path, "wb") as file:
                pickle.dump(scaler, file)

    def classify(self, parameters: Dict = None, **kwargs) -> np.array:
        """
        Classification method used to classify emotions from speech

        :param parameters: Parameter dictionary used for classification
        :param kwargs: Additional kwargs parameters
        :return: An array with predicted emotion indices
        """
        parameters = self.init_parameters(parameters, **kwargs)
        which_set = parameters.get("which_set", Set.TEST)
        dataset = self.data_reader.get_emotion_data(
            self.emotions, which_set, -1, parameters
        )
        features = np.empty((0, parameters.get("mfcc_num", 40) + 4))
        for data, class_name in dataset:
            for index, sample in enumerate(data):
                sample_features = np.reshape(
                    self.get_mixed_features(sample, parameters), (1, -1)
                )
                sample_features = self.scaler[class_name].transform(
                    sample_features
                )
                features = np.concatenate([features, sample_features], axis=0)

        predictions = np.zeros((features.shape[0], 7))
        for index, class_name in enumerate(CLASS_NAMES):
            predictions[:, index] = self.models[class_name].score_samples(
                features
            )
        return np.argmax(predictions, axis=1)


if __name__ == "__main__":  # pragma: no cover
    classifier = GMMClassifier()
    classifier.train({"which_set": Set.VAL})
    classifier.save()
    classifier.load()
    emotions = classifier.classify()
    labels = classifier.data_reader.get_labels(Set.TEST)
    print(f"Labels Shape: {labels.shape}")
    print(f"Emotions Shape: {emotions.shape}")
    print(f"Accuracy: {np.sum(emotions == labels) / labels.shape[0]}")