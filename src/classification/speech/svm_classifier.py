""" This file contains the MFCC-LSTM speech emotion classifier """
import os
import pickle
from typing import Dict

import numpy as np
from alive_progress import alive_bar
from sklearn import preprocessing, svm
from sklearn.multiclass import OneVsOneClassifier

from src.classification.speech.speech_emotion_classifier import (
    SpeechEmotionClassifier,
)
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


class SVMClassifier(SpeechEmotionClassifier):
    """
    Class that implements a speech emotion classifier that uses
    a Support Vector Machine for classification
    """

    def __init__(self, parameters: Dict = None):
        """
        Initialize the SVM emotion classifier

        :param name: The name for the classifier
        :param parameters: Some configuration parameters for the classifier
        """
        super().__init__("svm", parameters)
        self.model = None
        self.scaler = None

    def train(self, parameters: Dict = None, **kwargs) -> None:
        """
        Training method for SVM model

        :param parameters: Parameter dictionary used for training
        :param kwargs: Additional kwargs parameters
        """
        parameters = self.init_parameters(parameters, **kwargs)
        which_set = parameters.get("which_set", Set.TRAIN)
        batch_size = parameters.get("batch_size", 64)
        kernel = parameters.get("kernel", "rbf")
        self.train_data = self.data_reader.get_emotion_data(
            self.emotions, which_set, batch_size, parameters
        )
        features = np.empty((0, parameters.get("mfcc_num", 40) + 4))
        labels = np.empty((0, 1))
        batches = self.data_reader.num_batch[which_set]
        with alive_bar(
            batches, title="Processing batches", force_tty=True
        ) as bar:
            for data, batch_labels in self.train_data:
                for index, sample in enumerate(data.numpy()):
                    sample_features = np.reshape(
                        self.get_mixed_features(sample, parameters), (1, -1)
                    )
                    features = np.concatenate(
                        [features, sample_features], axis=0
                    )
                labels = np.concatenate(
                    [
                        labels,
                        np.expand_dims(np.argmax(batch_labels, axis=1), 1),
                    ],
                    axis=0,
                )
                bar()
        self.scaler = preprocessing.StandardScaler().fit(features)
        tr_fea = self.scaler.transform(features)

        self.model = (
            OneVsOneClassifier(
                svm.SVC(
                    kernel=kernel,
                    gamma="auto",
                    C=1,
                    class_weight="balanced",
                    tol=1e-15,
                    verbose=True,
                    max_iter=100000,
                )
            )
        ).fit(tr_fea, labels)

        self.is_trained = True

    def load(self, parameters: Dict = None, **kwargs) -> None:
        """
        Loading method that loads a previously trained model from disk.

        :param parameters: Parameters required for loading the model
        :param kwargs: Additional kwargs parameters
        """
        parameters = self.init_parameters(parameters, **kwargs)
        save_path = parameters.get("save_path", "models/speech/svm")
        model_path = os.path.join(save_path, "model.pkl")
        with open(model_path, "rb") as file:
            self.model = pickle.load(file)
        scaler_path = os.path.join(save_path, "scaler.pkl")
        with open(scaler_path, "rb") as file:
            self.scaler = pickle.load(file)
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
        save_path = parameters.get("save_path", "models/speech/svm")
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        model_path = os.path.join(save_path, "model.pkl")
        with open(model_path, "wb") as file:
            pickle.dump(self.model, file)
        scaler_path = os.path.join(save_path, "scaler.pkl")
        with open(scaler_path, "wb") as file:
            pickle.dump(self.scaler, file)

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
        features = np.empty((0, parameters.get("mfcc_num", 40) + 4))
        labels = np.empty((0, 1))
        batches = self.data_reader.num_batch[which_set]
        with alive_bar(
            batches, title="Processing batches", force_tty=True
        ) as bar:
            for data, batch_labels in dataset:
                for index, sample in enumerate(data.numpy()):
                    sample_features = np.reshape(
                        self.get_mixed_features(sample, parameters), (1, -1)
                    )
                    features = np.concatenate(
                        [features, sample_features], axis=0
                    )
                labels = np.concatenate(
                    [
                        labels,
                        np.expand_dims(np.argmax(batch_labels, axis=1), 1),
                    ],
                    axis=0,
                )
                bar()
        tr_features = self.scaler.transform(features)
        predictions = self.model.predict(tr_features)
        return predictions


if __name__ == "__main__":  # pragma: no cover
    classifier = SVMClassifier()
    classifier.train({"shuffle": True})
    classifier.save()
    classifier.load()
    emotions = classifier.classify()
    labels = classifier.data_reader.get_labels(Set.TEST)
    print(f"Labels Shape: {labels.shape}")
    print(f"Emotions Shape: {emotions.shape}")
    print(f"Accuracy: {np.sum(emotions == labels) / labels.shape[0]}")
