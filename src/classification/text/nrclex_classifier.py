"""Implementation of a text classifier using the NRCLex python library"""
from typing import Dict

import numpy as np
from nrclex import NRCLex

from src.classification.text.text_emotion_classifier import (
    TextEmotionClassifier,
)
from src.data.data_reader import Set


class NRCLexTextClassifier(TextEmotionClassifier):
    """
    This class implements a text classifier using the NRCLex python library.
    It uses a lexicon for finding emotion words that are then used to make
    a classification. This means that there is no training necessary.
    """

    def __init__(self, parameters: Dict = None):
        """
        Initialize the NRCLex classifier

        :param parameters: Parameters used to configure the classifier
        """
        super().__init__(name="nrclex", parameters=parameters)
        self.emotion_map = {
            "fear": 4,
            "anger": 0,
            "anticipation": 3,
            "surprise": 1,
            "sadness": 5,
            "disgust": 2,
            "joy": 3,
        }

    def train(self, parameters: Dict = None, **kwargs) -> None:
        """
        Training method for the classifier. The classifier does not require
        training as it is only a lexicon-based method.

        :param parameters: parameters for training
        :param kwargs: additional parameters
        """
        pass

    def load(self, parameters: Dict = None, **kwargs) -> None:
        """
        Load a stored classifier from storage. Not necessary for NRCLex.

        :param parameters: Loading parameters
        :param kwargs: Additional parameters
        """
        pass

    def save(self, parameters: Dict = None, **kwargs) -> None:
        """
        Save a trained classifier in storage. Not necessary for NRCLex.

        :param parameters: Saving parameters
        :param kwargs: Additional parameters
        """
        pass

    def classify(self, parameters: Dict = None, **kwargs) -> np.array:
        """
        Classify the emotions based on the NRCLex library.

        :param parameters: Classification parameters
        :param kwargs: Additional parameters
        :return: An array of the classification results, shape (num_samples,)
        """
        parameters = parameters or {}
        which_set = parameters.get("set", Set.TEST)
        batch_size = parameters.get("batch_size", 64)
        dataset = self.data_reader.get_emotion_data(
            self.emotions, which_set, batch_size, shuffle=False
        )
        results = []
        for texts, _ in dataset:
            texts = list(texts.numpy())
            for text in texts:
                emotion = NRCLex(text[0].decode())
                results.append(
                    self.get_best_emotion(emotion.raw_emotion_scores)
                )
        return np.asarray(results)

    def get_best_emotion(self, raw_scores: Dict) -> int:
        """
        Gets the emotion scores from the NRCLex library and then gets the most
        likely emotion from that.

        :param raw_scores: The raw scores dictionary generated by NRCLex
        :return: The integer index of the emotion in the neutral_ekman space
        """
        emotion = {}
        for key, value in raw_scores.items():
            if key in self.emotion_map:
                emotion_int = self.emotion_map[key]
                if emotion_int not in emotion:
                    emotion[emotion_int] = 0
                emotion[emotion_int] += value
        positives = emotion.get(3, 0) + emotion.get(1, 0)
        negatives = np.sum([emotion.get(key, 0) for key in [0, 2, 4, 5]])

        if positives == negatives:
            return 6  # neutral

        if positives > negatives:
            keys = np.array([3, 1])
        else:
            keys = np.array([0, 2, 4, 5])
        values = np.array([emotion.get(key, 0) for key in keys])
        max_keys = keys[values == np.max(values)]
        return np.random.choice(max_keys)


if __name__ == "__main__":  # pragma: no cover
    classifier = NRCLexTextClassifier()
    emotions = classifier.classify()
    labels = classifier.data_reader.get_labels(Set.TEST)
    print(f"Labels Shape: {labels.shape}")
    print(f"Emotions Shape: {emotions.shape}")
    print(f"Accuracy: {np.sum(emotions == labels) / labels.shape[0]}")
