"""Implements a text emotion classifier based on Distilbert"""
from typing import Dict

import numpy as np
import tensorflow as tf

from src.classification.text.bert_classifier import BertClassifier
from src.data.data_reader import Set


class DistilBertClassifier(BertClassifier):
    """
    This class implements the Distilbert model for emotion classification.
    We can reuse the training and classification functionality from BERT.
    """

    def __init__(self, parameters: Dict = None):
        """
        Initialize the emotion classifier.

        :param parameters: Configuration parameters
        """
        super().__init__("distilbert", parameters)
        self.model_name = "distilbert_en_uncased_L-6_H-768_A-12"
        self.model_path = f"https://tfhub.dev/jeongukjae/{self.model_name}/1"
        self.preprocess_path = (
            "https://tfhub.dev/jeongukjae/distilbert_en_uncased_preprocess/2"
        )

    def load(self, parameters: Dict = None, **kwargs) -> None:
        """
        Loading method for the BERT classifier that loads a stored model from
        disk.

        :param parameters: Loading parameters
        :param kwargs: Additional parameters
        """
        parameters = parameters or {}
        save_path = parameters.get("save_path", "models/text/distilbert")
        self.classifier = tf.keras.models.load_model(save_path)

    def save(self, parameters: Dict = None, **kwargs) -> None:
        """
        Saving method for the BERT classifier that saves a trained model from
        disk.

        :param parameters: Saving parameters
        :param kwargs: Additional parameters
        """
        if not self.is_trained:
            raise RuntimeError(
                "Model needs to be trained in order to save it!"
            )
        parameters = parameters or {}
        save_path = parameters.get("save_path", "models/text/distilbert")
        self.classifier.save(save_path, include_optimizer=False)


if __name__ == "__main__":  # pragma: no cover
    classifier = DistilBertClassifier()
    classifier.train({"epochs": 10, "dropout_rate": 0.2})
    classifier.save()
    classifier.load()
    emotions = classifier.classify()
    labels = classifier.data_reader.get_labels(Set.TEST)
    print(f"Labels Shape: {labels.shape}")
    print(f"Emotions Shape: {emotions.shape}")
    print(f"Accuracy: {np.sum(emotions == labels) / labels.shape[0]}")
