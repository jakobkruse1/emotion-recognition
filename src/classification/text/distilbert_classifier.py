"""Implements a text emotion classifier based on Distilbert"""

from typing import Dict

import tensorflow as tf

from src.classification.text.bert_classifier import BertClassifier
from src.utils import logging, training_loop


class DistilBertClassifier(BertClassifier):
    """
    This class implements the Distilbert model for emotion classification.
    We can reuse the training and classification functionality from BERT.
    """

    def __init__(self, parameters: Dict = None):
        """
        Initialize the emotion classifier.

        :param parameters: Configuration parameters
            Not currently used
        """
        super().__init__(parameters)
        self.name = "distilbert"
        self.model_name = "distilbert_en_uncased_L-6_H-768_A-12"
        self.model_path = f"https://tfhub.dev/jeongukjae/{self.model_name}/1"
        self.preprocess_path = (
            "https://tfhub.dev/jeongukjae/distilbert_en_uncased_preprocess/2"
        )
        self.logger = logging.KerasLogger()
        self.logger.log_start(
            {"init_parameters": parameters, "model_name": self.model_name}
        )

    def load(self, parameters: Dict = None, **kwargs) -> None:
        """
        Loading method for the classifier that loads a stored model from
        disk.

        :param parameters: Loading parameters
            save_path: Folder where model is loaded from
        :param kwargs: Additional parameters
            Not used currently
        """
        parameters = self.init_parameters(parameters, **kwargs)
        save_path = parameters.get("save_path", "models/text/distilbert")
        self.classifier = tf.keras.models.load_model(save_path)

    def save(self, parameters: Dict = None, **kwargs) -> None:
        """
        Saving method for the classifier that saves a trained model from
        disk.

        :param parameters: Saving parameters
            save_path: Folder where model is saved at
        :param kwargs: Additional parameters
            Not used currently
        """
        if not self.is_trained:
            raise RuntimeError(
                "Model needs to be trained in order to save it!"
            )
        parameters = self.init_parameters(parameters, **kwargs)
        save_path = parameters.get("save_path", "models/text/distilbert")
        self.classifier.save(save_path, include_optimizer=False)
        self.logger.log_start({"train_parameters": parameters})


def _main():  # pragma: no cover
    classifier = DistilBertClassifier()
    parameters = {"init_lr": 1e-05, "dropout_rate": 0.2, "dense_layer": 1024}
    save_path = "models/text/distilbert"
    training_loop(classifier, parameters, save_path)


if __name__ == "__main__":  # pragma: no cover
    _main()
