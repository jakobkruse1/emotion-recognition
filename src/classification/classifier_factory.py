"""Factory for emotion classifiers"""
from typing import Dict

from src.classification.emotion_classifier import EmotionClassifier
from src.classification.image import (
    CrossAttentionNetworkClassifier,
    MultiTaskEfficientNetB2Classifier,
    VGG16Classifier,
)
from src.classification.plant import (
    PlantDenseClassifier,
    PlantLSTMClassifier,
    PlantMFCCCNNClassifier,
    PlantMFCCResnetClassifier,
)
from src.classification.speech import (
    BYOLSClassifier,
    GMMClassifier,
    HMMClassifier,
    HuBERTClassifier,
    MFCCLSTMClassifier,
    SVMClassifier,
    Wav2Vec2Classifier,
)
from src.classification.text import (
    BertClassifier,
    DistilBertClassifier,
    NRCLexTextClassifier,
)
from src.classification.watch import (
    WatchDenseClassifier,
    WatchLSTMClassifier,
    WatchRandomForestClassifier,
    WatchTransformerClassifier,
    WatchXGBoostClassifier,
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
        elif modality == "speech":
            return SpeechClassifierFactory.get(model, parameters)
        elif modality == "plant":
            return PlantClassifierFactory.get(model, parameters)
        elif modality == "watch":
            return WatchClassifierFactory.get(model, parameters)
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


class SpeechClassifierFactory:
    """
    Factory class that generates speech emotion classifiers
    """

    @staticmethod
    def get(model: str, parameters: Dict = None) -> EmotionClassifier:
        """
        Method that returns an instance of a speech emotion classifier

        :param model: The name of the speech model
        :param parameters: The parameters for the speech model
        :return: The constructed speech classifier
        """
        if model == "mfcc_lstm":
            return MFCCLSTMClassifier(parameters)
        elif model == "hubert":
            return HuBERTClassifier(parameters)
        elif model == "wav2vec2":
            return Wav2Vec2Classifier(parameters)
        elif model == "hmm":
            return HMMClassifier(parameters)
        elif model == "gmm":
            return GMMClassifier(parameters)
        elif model == "svm":
            return SVMClassifier(parameters)
        elif model == "byols":
            return BYOLSClassifier(parameters)
        else:
            raise ValueError(f"Speech model {model} not implemented!")


class PlantClassifierFactory:
    """
    Factory class that generates plant emotion classifiers
    """

    @staticmethod
    def get(model: str, parameters: Dict = None) -> EmotionClassifier:
        """
        Method that returns an instance of a plant emotion classifier

        :param model: The name of the plant model
        :param parameters: The parameters for the plant model
        :return: The constructed plant classifier
        """
        if model == "plant_lstm":
            return PlantLSTMClassifier(parameters)
        elif model == "plant_dense":
            return PlantDenseClassifier(parameters)
        elif model == "plant_mfcc_cnn":
            return PlantMFCCCNNClassifier(parameters)
        elif model == "plant_mfcc_resnet":
            return PlantMFCCResnetClassifier(parameters)
        else:
            raise ValueError(f"Plant model {model} not implemented!")


class WatchClassifierFactory:
    """
    Factory class that generates watch emotion classifiers
    """

    @staticmethod
    def get(model: str, parameters: Dict = None) -> EmotionClassifier:
        """
        Method that returns an instance of a watch emotion classifier

        :param model: The name of the watch model
        :param parameters: The parameters for the watch model
        :return: The constructed watch classifier
        """
        if model == "watch_lstm":
            return WatchLSTMClassifier(parameters)
        elif model == "watch_dense":
            return WatchDenseClassifier(parameters)
        elif model == "watch_random_forest":
            return WatchRandomForestClassifier(parameters)
        elif model == "watch_xgboost":
            return WatchXGBoostClassifier(parameters)
        elif model == "watch_transformer":
            return WatchTransformerClassifier(parameters)
        else:
            raise ValueError(f"Watch model {model} not implemented!")
