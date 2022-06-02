""" This file contains the MFCC-LSTM speech emotion classifier """
import os
import pickle
from typing import Dict

import numpy as np
import tensorflow as tf
from hmmlearn import hmm

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


class HMMClassifier(SpeechEmotionClassifier):
    """
    Class that implements a speech emotion classifier that uses
    a Hidden Markov Model for classification
    """

    def __init__(self, parameters: Dict = None):
        """
        Initialize the HMMemotion classifier

        :param name: The name for the classifier
        :param parameters: Some configuration parameters for the classifier
        """
        super().__init__("hmm", parameters)
        self.models = {}
        self.train_data_reader = ClasswiseSpeechDataReader(
            folder=self.data_reader.folder
        )
        input = tf.keras.layers.Input(
            shape=(48000), dtype=tf.float32, name="raw"
        )
        mfcc = self.compute_mfccs(input)
        self.mfcc_model = tf.keras.Model(input, mfcc)

    def train(self, parameters: Dict = None, **kwargs) -> None:
        """
        Training method for HMM model

        :param parameters: Parameter dictionary used for training
        :param kwargs: Additional kwargs parameters
        """
        parameters = self.init_parameters(parameters, **kwargs)
        which_set = parameters.get("which_set", Set.TRAIN)
        self.train_data = self.train_data_reader.get_emotion_data(
            self.emotions, which_set, -1, parameters
        )
        n_components = parameters.get("n_components", 4)

        for data, class_name in self.train_data:
            data = self.mfcc_model(data).numpy()
            model = hmm.GaussianHMM(
                n_components=n_components,
            )
            model.fit(
                np.reshape(
                    data, (data.shape[0] * data.shape[1], data.shape[2])
                )
            )
            self.models[class_name] = model

        self.is_trained = True

    def load(self, parameters: Dict = None, **kwargs) -> None:
        """
        Loading method that loads a previously trained model from disk.

        :param parameters: Parameters required for loading the model
        :param kwargs: Additional kwargs parameters
        """
        parameters = self.init_parameters(parameters, **kwargs)
        save_path = parameters.get("save_path", "models/speech/hmm")
        for class_name in CLASS_NAMES:
            model_path = os.path.join(save_path, f"{class_name}.pkl")
            with open(model_path, "rb") as file:
                model = pickle.load(file)
            self.models[class_name] = model
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
        save_path = parameters.get("save_path", "models/speech/hmm")
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        for name, model in self.models.items():
            model_path = os.path.join(save_path, f"{name}.pkl")
            with open(model_path, "wb") as file:
                pickle.dump(model, file)

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
        predictions = []
        for data, labels in dataset:
            data = self.mfcc_model(data).numpy()
            for sample in data:
                class_prediction = []
                for class_name in CLASS_NAMES:
                    sample_prediction = self.models[class_name].score(sample)
                    class_prediction.append(sample_prediction)
                predictions.append(np.argmax(class_prediction))
        predictions = np.asarray(predictions)
        return predictions


if __name__ == "__main__":  # pragma: no cover
    classifier = HMMClassifier()
    classifier.train({"which_set": Set.VAL})
    classifier.save()
    classifier.load()
    emotions = classifier.classify()
    labels = classifier.data_reader.get_labels(Set.TEST)
    print(f"Labels Shape: {labels.shape}")
    print(f"Emotions Shape: {emotions.shape}")
    print(f"Accuracy: {np.sum(emotions == labels) / labels.shape[0]}")
