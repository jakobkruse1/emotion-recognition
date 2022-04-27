""" This file contains the EfficientNet facial emotion classifier """

from typing import Dict

import numpy as np
import tensorflow as tf

from src.classification.image.image_emotion_classifier import (
    ImageEmotionClassifier,
)
from src.data.data_reader import Set


class VGG16Classifier(ImageEmotionClassifier):
    """
    Class that implements an emotion classifier using VGG16
    """

    def __init__(self, name: str = "vgg16", parameters: Dict = None) -> None:
        """
        Initialize the VGG16 emotion classifier

        :param name: The name for the classifier
        :param parameters: Some configuration parameters for the classifier
        """
        super().__init__(name, parameters)
        tf.get_logger().setLevel("ERROR")
        self.model = None

    def initialize_model(self) -> None:
        """
        Initializes a new and pretrained version of the VGG16Classifier model
        """
        input = tf.keras.layers.Input(
            shape=(48, 48, 3), dtype=tf.float32, name="image"
        )
        model = tf.keras.applications.VGG16(
            include_top=False,
            weights="imagenet",
            input_tensor=input,
            input_shape=(48, 48, 3),
        )
        for layer in model.layers[:-4]:
            layer.trainable = False
        out = model(input)

        out = tf.keras.layers.Flatten()(out)
        out = tf.keras.layers.Dense(4096, activation="relu")(out)
        out = tf.keras.layers.Dense(4096, activation="relu")(out)

        top = tf.keras.layers.Dense(
            7, activation="softmax", name="classifier"
        )(out)
        self.model = tf.keras.Model(input, top)

    def train(self, parameters: Dict = None, **kwargs) -> None:
        """
        Training method for VGG16Classifier model

        :param parameters: Parameter dictionary used for training
        :param kwargs: Additional kwargs parameters
        """
        parameters = parameters or {}
        epochs = parameters.get("epochs", 20)
        which_set = parameters.get("which_set", Set.TRAIN)
        batch_size = parameters.get("batch_size", 64)
        loss = tf.keras.losses.CategoricalCrossentropy()
        metrics = [tf.metrics.CategoricalAccuracy()]

        if not self.model:
            self.initialize_model()
        callback = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=3
        )
        self.model.compile(optimizer="adam", loss=loss, metrics=metrics)
        train_data = self.data_reader.get_emotion_data(
            self.emotions, which_set, batch_size
        ).map(lambda x, y: (tf.image.grayscale_to_rgb(x), y))
        val_data = self.data_reader.get_emotion_data(
            self.emotions, Set.VAL, batch_size
        ).map(lambda x, y: (tf.image.grayscale_to_rgb(x), y))

        _ = self.model.fit(
            x=train_data,
            validation_data=val_data,
            epochs=epochs,
            callbacks=[callback],
        )
        self.is_trained = True

    def load(self, parameters: Dict = None, **kwargs) -> None:
        """
        Loading method that loads a previously trained model from disk.

        :param parameters: Parameters required for loading the model
        :param kwargs: Additional kwargs parameters
        """
        parameters = parameters or {}
        save_path = parameters.get("save_path", "models/image/cross_attention")
        self.model = tf.keras.models.load_model(save_path)

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
        parameters = parameters or {}
        save_path = parameters.get("save_path", "models/image/cross_attention")
        self.model.save(save_path, include_optimizer=False)

    def classify(self, parameters: Dict = None, **kwargs) -> np.array:
        """
        Classification method used to classify emotions from images

        :param parameters: Parameter dictionary used for classification
        :param kwargs: Additional kwargs parameters
        :return: An array with predicted emotion indices
        """
        parameters = parameters or {}
        which_set = parameters.get("which_set", Set.TEST)
        batch_size = parameters.get("batch_size", 64)
        dataset = self.data_reader.get_emotion_data(
            self.emotions, which_set, batch_size, shuffle=False
        ).map(lambda x, y: (tf.image.grayscale_to_rgb(x), y))

        if not self.model:
            raise RuntimeError(
                "Please load or train the model before inference!"
            )
        results = self.model.predict(dataset)
        return np.argmax(results, axis=1)


if __name__ == "__main__":  # pragma: no cover
    classifier = VGG16Classifier()
    classifier.train({"epochs": 1})
    classifier.save()
    classifier.load()
    emotions = classifier.classify()
    labels = classifier.data_reader.get_labels(Set.TEST)
    print(f"Labels Shape: {labels.shape}")
    print(f"Emotions Shape: {emotions.shape}")
    print(f"Accuracy: {np.sum(emotions == labels) / labels.shape[0]}")
