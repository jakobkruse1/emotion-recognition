""" This file defines the emotion sets that are available in this work. """

from typing import Iterable, Union

import numpy as np


class EmotionMapper:
    def __init__(self):
        """
        Class that maps more complex emotions down to the NeutralEkman emotions
        """
        self.emotion_map = {}
        self.setup_map()

    def map_emotion(self, emotion: str) -> str:
        """
        Maps an emotion to its equivalent in the NeutralEkman emotion set.

        :param emotion: The name of the emotion to map to Ekman
        :return: The name of the corresponding Ekman emotion
        """
        return self.emotion_map[emotion]

    def setup_map(self) -> None:
        """
        One time setup for the emotion map in the class.
        """
        for emotion in ["anger", "annoyance", "disapproval"]:
            self.emotion_map[emotion] = "anger"
        for emotion in ["disgust"]:
            self.emotion_map[emotion] = "disgust"
        for emotion in ["fear", "nervousness"]:
            self.emotion_map[emotion] = "fear"
        for emotion in [
            "joy",
            "amusement",
            "approval",
            "excitement",
            "gratitude",
            "love",
            "optimism",
            "relief",
            "pride",
            "admiration",
            "desire",
            "caring",
        ]:
            self.emotion_map[emotion] = "joy"
        for emotion in [
            "sadness",
            "disappointment",
            "embarrassment",
            "grief",
            "remorse",
        ]:
            self.emotion_map[emotion] = "sadness"
        for emotion in ["surprise", "realization", "confusion", "curiosity"]:
            self.emotion_map[emotion] = "surprise"
        for emotion in ["neutral"]:
            self.emotion_map[emotion] = "neutral"


class AbstractEmotionSet:
    """
    This class implements the abstract interface for all emotion sets.

    One problem in emotion detection is that a lot of different emotion sets
    are used throughout the past research. While one paper predicts only
    positive, neutral and negative emotions, another one might differentiate
    between 6 or more distinct emotions (joy, anger, fear, ...).
    This class models this behaviour by defining different emotion sets.
    """

    def __init__(self, name: str, count: int, classes: Iterable[str]):
        """
        Initialize an emotion set with the required information

        :param name: A name for the emotion set to distinguish them
        :param count: The number of distinct emotions used
        :param classes: The names of the distinct emotions in a List
        """
        self.name = name
        self.emotion_count = count
        self.emotion_names = classes

    def get_emotions(
        self, indices: Union[int, np.ndarray]
    ) -> Union[str, np.ndarray]:
        """
        This function returns the emotion strings for given indices

        :param indices: The index or indices of the emotions
        :return: Array of emotion strings or single emotion string
        """
        assert np.logical_and(indices >= 0, indices < self.emotion_count).all()
        return np.array(self.emotion_names)[indices]


class ThreeEmotions(AbstractEmotionSet):
    """
    Simple emotion set with only three categories: positive, neutral, negative
    """

    def __init__(self):
        """
        Initializer for the emotion set
        """
        super().__init__("three", 3, ["positive", "neutral", "negative"])


class EkmanEmotions(AbstractEmotionSet):
    """
    Emotion set defined by Paul Ekman that is commonly used in research.
    It contains 6 basic emotions:
    anger, surprise, disgust, enjoyment, fear, and sadness
    """

    def __init__(self):
        """
        Initializer for the Ekman emotion set
        """
        super().__init__(
            "ekman",
            6,
            ["anger", "surprise", "disgust", "joy", "fear", "sadness"],
        )


class EkmanNeutralEmotions(AbstractEmotionSet):
    """
    Ekman Emotion set extended by a neutral state.
    """

    def __init__(self):
        """
        Initializer for the emotion set
        """
        super().__init__(
            "neutral_ekman",
            7,
            [
                "anger",
                "surprise",
                "disgust",
                "joy",
                "fear",
                "sadness",
                "neutral",
            ],
        )


class EmotionSetFactory:
    """
    Factory class that generates emotion set instances
    """

    @staticmethod
    def generate(name: str) -> AbstractEmotionSet:
        """
        Method that creates and returns an instance of the emotion set
        specified by the name parameter

        :param name: The name of the desired emotion set
        :raise ValueError: Raised if the name does not represent an emotion set
        :return: The created emotion set
        """
        if name == "three":
            return ThreeEmotions()
        elif name == "ekman":
            return EkmanEmotions()
        elif name == "neutral_ekman":
            return EkmanNeutralEmotions()
        else:
            raise ValueError(
                f'The emotion set with name "{name}" does not exist!'
            )
