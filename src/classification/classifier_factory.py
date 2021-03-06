"""Factory for emotion classifiers"""
from typing import Dict

from src.classification.emotion_classifier import EmotionClassifier
from src.classification.image import (
    CrossAttentionNetworkClassifier,
    MultiTaskEfficientNetB2Classifier,
    VGG16Classifier,
)
from src.classification.text import (
    BertClassifier,
    DistilBertClassifier,
    NRCLexTextClassifier,
)


class ClassifierFactory:
    """
    Factory class that generates emotion classifiers
    """

    @staticmethod
    def get(
        modality: str, model: str, parameters: Dict = None
    ) -> EmotionClassifier:
        """
        Creates an instance of an emotion classifier

        :param modality: The modality of the classifier
        :param model: The model name for the classifier
        :param parameters: The parameters for the classifier
        :return: The constructed emotion classifier
        """
        if modality == "text":
            return TextClassifierFactory.get(model, parameters)
        elif modality == "image":
            return ImageClassifierFactory.get(model, parameters)
        else:
            raise ValueError(f"Modality {modality} not supported!")


class TextClassifierFactory:
    """
    Factory class that generates text emotion classifiers
    """

    @staticmethod
    def get(model: str, parameters: Dict = None) -> EmotionClassifier:
        """
        Method that returns an instance of a text emotion classifier

        :param model: The name of the text model
        :param parameters: The parameters for the text model
        :return: The constructed text classifier
        """
        if model == "nrclex":
            return NRCLexTextClassifier(parameters)
        elif model == "bert":
            return BertClassifier(parameters)
        elif model == "distilbert":
            return DistilBertClassifier(parameters)
        else:
            raise ValueError(f"Text model {model} not implemented!")


class ImageClassifierFactory:
    """
    Factory class that generates image emotion classifiers
    """

    @staticmethod
    def get(model: str, parameters: Dict = None) -> EmotionClassifier:
        """
        Method that returns an instance of an image emotion classifier

        :param model: The name of the image model
        :param parameters: The parameters for the image model
        :return: The constructed image classifier
        """
        if model == "efficientnet":
            return MultiTaskEfficientNetB2Classifier(parameters)
        elif model == "vgg16":
            return VGG16Classifier(parameters)
        elif model == "cross_attention":
            return CrossAttentionNetworkClassifier(parameters)
        else:
            raise ValueError(f"Image model {model} not implemented!")
