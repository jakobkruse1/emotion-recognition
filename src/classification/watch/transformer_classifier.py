""" This file implements an LSTM based classifier for the watch data. """

from typing import Dict

import tensorflow as tf

from src.classification.watch.nn_classifier import WatchNNBaseClassifier
from src.utils import cv_training_loop


class WatchTransformerClassifier(WatchNNBaseClassifier):
    """
    Classifier that uses Transformer layers and a Dense head for classification.
    """

    def __init__(self, parameters: Dict = None):
        """
        Initialize the Watch-Transformer emotion classifier

        :param parameters: Some configuration parameters for the classifier
        """
        super().__init__("watch_transformer", parameters)

    def initialize_model(self, parameters: Dict) -> None:
        """
        Initializes a new and pretrained version of the Watch-Transformer model

        :param parameters: Parameters for initializing the model.
        """
        num_transformer_blocks = parameters.get("num_transformer_blocks", 6)
        head_size = parameters.get("head_size", 64)
        num_heads = parameters.get("num_heads", 8)
        ff_dim = parameters.get("ff_dim", 512)
        dropout = parameters.get("dropout", 0.2)
        dense_layers = parameters.get("dense_layers", 2)
        dense_units = parameters.get("dense_units", 1024)

        def transformer_layer(inputs, head_size, num_heads, ff_dim, dropout=0):
            # Normalization and Attention
            x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(inputs)
            x = tf.keras.layers.MultiHeadAttention(
                key_dim=head_size, num_heads=num_heads, dropout=dropout
            )(x, x)
            x = tf.keras.layers.Dropout(dropout)(x)
            res = x + inputs

            # Feed Forward Part
            x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(res)
            x = tf.keras.layers.Conv1D(
                filters=ff_dim, kernel_size=1, activation="relu"
            )(x)
            x = tf.keras.layers.Dropout(dropout)(x)
            x = tf.keras.layers.Conv1D(
                filters=inputs.shape[-1], kernel_size=1
            )(x)
            return x + res

        input_size = self.data_reader.get_input_shape(parameters)
        input_tensor = tf.keras.layers.Input(
            shape=(*input_size,), dtype=tf.float32, name="raw"
        )
        x = input_tensor
        for _ in range(num_transformer_blocks):
            x = transformer_layer(x, head_size, num_heads, ff_dim, dropout)

        x = tf.keras.layers.GlobalAveragePooling1D(
            data_format="channels_first"
        )(x)
        for _ in range(dense_layers):
            x = tf.keras.layers.Dense(dense_units, activation="relu")(x)
            x = tf.keras.layers.Dropout(dropout)(x)
        out = tf.keras.layers.Dense(7, activation="softmax")(x)

        self.model = tf.keras.Model(input_tensor, out)


def _main():  # pragma: no cover
    classifier = WatchTransformerClassifier()
    parameters = {
        "epochs": 1000,
        "patience": 100,
        "batch_size": 64,
        "window": 20,
        "hop": 2,
        "balanced": True,
        "label_mode": "both",
        "ff_dim": 512,
        "dropout": 0.2,
        "dense_layers": 3,
        "dense_units": 1024,
    }
    save_path = "models/watch/watch_transformer"
    cv_training_loop(classifier, parameters, save_path)


if __name__ == "__main__":  # pragma: no cover
    _main()
