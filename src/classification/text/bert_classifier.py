"""Implementation of an emotion classifier using BERT"""
from typing import Dict

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text  # noqa: F401
from official.nlp import optimization

from src.classification.text.text_emotion_classifier import (
    TextEmotionClassifier,
)
from src.data.data_reader import Set


class BertClassifier(TextEmotionClassifier):
    """
    Emotion classifier based on the BERT model
    """

    def __init__(self, parameters: Dict = None):
        """
        Initialize the emotion classifier.

        :param name: The classifier name
        :param parameters: Configuration parameters
        """
        super().__init__("bert", parameters)
        tf.get_logger().setLevel("ERROR")
        parameters = parameters or {}
        self.model_name = parameters.get(
            "model_name", "bert_en_uncased_L-4_H-256_A-4"
        )
        self.model_path = (
            f"https://tfhub.dev/tensorflow/small_bert/" f"{self.model_name}/1"
        )
        self.preprocess_path = (
            "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3"
        )
        self.classifier = None

    def _build_classifier_model(self, parameters: Dict) -> tf.keras.Model:
        """
        Define the tensorflow model that uses BERT for classification

        :return: The keras model that classifies the text
        """
        dropout_rate = parameters.get("dropout_rate", 0.1)
        input = tf.keras.layers.Input(shape=(), dtype=tf.string, name="text")
        preprocessor = hub.KerasLayer(
            self.preprocess_path, name="preprocessing"
        )
        encoder_inputs = preprocessor(input)
        encoder = hub.KerasLayer(
            self.model_path, trainable=True, name="BERT_encoder"
        )
        outputs = encoder(encoder_inputs)
        net = outputs["pooled_output"]
        net = tf.keras.layers.Dropout(dropout_rate)(net)
        net = tf.keras.layers.Dense(
            7, activation="softmax", name="classifier"
        )(net)
        return tf.keras.Model(input, net)

    def train(self, parameters: Dict = None, **kwargs) -> None:
        """
        Training method for the BERT classifier

        :param parameters: Training parameters
        :param kwargs: Additional parameters
        """
        parameters = parameters or {}
        init_lr = parameters.get("init_lr", 3e-5)
        epochs = parameters.get("epochs", 20)
        which_set = parameters.get("set", Set.TRAIN)
        batch_size = parameters.get("batch_size", 64)

        num_samples = self.data_reader.get_labels(which_set).shape[0]
        num_train_steps = int(num_samples * epochs / batch_size)
        self.classifier = self._build_classifier_model(parameters)
        loss = tf.keras.losses.CategoricalCrossentropy()
        metrics = [tf.metrics.CategoricalAccuracy()]
        optimizer = optimization.create_optimizer(
            init_lr=init_lr,
            num_train_steps=num_train_steps,
            num_warmup_steps=0.1 * num_train_steps,
            optimizer_type="adamw",
        )
        callback = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=3
        )
        self.classifier.compile(
            optimizer=optimizer, loss=loss, metrics=metrics
        )

        train_data = self.data_reader.get_emotion_data(
            self.emotions, which_set, batch_size
        )
        val_data = self.data_reader.get_emotion_data(
            self.emotions, Set.VAL, batch_size
        )

        _ = self.classifier.fit(
            x=train_data,
            validation_data=val_data,
            epochs=epochs,
            callbacks=[callback],
        )
        self.is_trained = True

    def load(self, parameters: Dict = None, **kwargs) -> None:
        """
        Loading method for the BERT classifier that loads a stored model from
        disk.

        :param parameters: Loading parameters
        :param kwargs: Additional parameters
        """
        parameters = parameters or {}
        save_path = parameters.get("save_path", "models/text/bert")
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
        save_path = parameters.get("save_path", "models/text/bert")
        self.classifier.save(save_path, include_optimizer=False)

    def classify(self, parameters: Dict = None, **kwargs) -> np.array:
        """
        Classify method for the BERT classifier

        :param parameters: Loading parameters
        :param kwargs: Additional parameters
        """
        parameters = parameters or {}
        which_set = parameters.get("set", Set.TEST)
        batch_size = parameters.get("batch_size", 64)
        dataset = self.data_reader.get_emotion_data(
            self.emotions, which_set, batch_size, shuffle=False
        )
        if not self.classifier:
            raise RuntimeError(
                "Please load or train the model before inference!"
            )
        results = self.classifier.predict(dataset)
        return np.argmax(results, axis=1)


if __name__ == "__main__":  # pragma: no cover
    classifier = BertClassifier()
    classifier.train()
    classifier.save()
    classifier.load()
    emotions = classifier.classify()
    labels = classifier.data_reader.get_labels(Set.TEST)
    print(f"Labels Shape: {labels.shape}")
    print(f"Emotions Shape: {emotions.shape}")
    print(f"Accuracy: {np.sum(emotions == labels) / labels.shape[0]}")
