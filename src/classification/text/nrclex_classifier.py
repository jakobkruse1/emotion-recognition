"""Implementation of a text classifier using the NRCLex python library"""
from typing import Dict

import numpy as np

from src.classification.text.text_emotion_classifier import (
    TextEmotionClassifier,
)


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
        :return: An array of the classification results
        """
        pass
