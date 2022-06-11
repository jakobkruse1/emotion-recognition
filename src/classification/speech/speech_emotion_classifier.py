""" Base class for all speech emotion classifiers """

from abc import abstractmethod
from typing import Dict

import numpy as np
import tensorflow as tf

from src.classification import EmotionClassifier
from src.data.data_reader import Set


class SpeechEmotionClassifier(EmotionClassifier):
    """
    Base class for all speech emotion classifiers. Contains common functions
    that concerns all speech classifiers.
    """

    def __init__(self, name: str = "speech", parameters: Dict = None):
        """
        Initialize the Speech emotion classifier

        :param name: The name for the classifier
        :param parameters: Some configuration parameters for the classifier
        """
        super().__init__(name, "speech", parameters)
        parameters = parameters or {}
        self.callback = None
        self.optimizer = None
        self.loss = None
        self.metrics = None
        self.train_data = None
        self.val_data = None
        self.class_weights = None

    @abstractmethod
    def train(self, parameters: Dict = None, **kwargs) -> None:
        """
        Virtual training method for interfacing

        :param parameters: Parameter dictionary used for training
        :param kwargs: Additional kwargs parameters
        """
        raise NotImplementedError("Abstract class")  # pragma: no cover

    @abstractmethod
    def load(self, parameters: Dict = None, **kwargs) -> None:
        """
        Loading method that loads a previously trained model from disk.

        :param parameters: Parameters required for loading the model
        :param kwargs: Additional kwargs parameters
        """
        raise NotImplementedError("Abstract class")  # pragma: no cover

    @abstractmethod
    def save(self, parameters: Dict = None, **kwargs) -> None:
        """
        Saving method that saves a previously trained model on disk.

        :param parameters: Parameters required for storing the model
        :param kwargs: Additional kwargs parameters
        """
        raise NotImplementedError("Abstract class")  # pragma: no cover

    @abstractmethod
    def classify(self, parameters: Dict = None, **kwargs) -> np.array:
        """
        The virtual classification method for interfacing

        :param parameters: Parameter dictionary used for classification
        :param kwargs: Additional kwargs parameters
        :return: An array with predicted emotion indices
        """
        raise NotImplementedError("Abstract class")  # pragma: no cover

    def prepare_training(self, parameters: Dict) -> None:
        """
        Function that prepares the training by initializing optimizer,
        loss, metrics and callbacks for training.

        :param parameters: Training parameters
        """
        learning_rate = parameters.get("learning_rate", 0.001)
        patience = parameters.get("patience", 5)

        self.callback = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=patience, restore_best_weights=True
        )
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.metrics = [tf.metrics.CategoricalAccuracy()]
        self.loss = tf.keras.losses.CategoricalCrossentropy()

    def prepare_data(self, parameters: Dict) -> None:
        """
        Function that prepares speech datasets for training and stores them
        inside the class.

        :param parameters: Parameter dictionary that contains important params.
            including: which_set, batch_size, weighted
        """
        which_set = parameters.get("which_set", Set.TRAIN)
        batch_size = parameters.get("batch_size", 64)
        weighted = parameters.get("weighted", False)

        self.train_data = self.data_reader.get_emotion_data(
            self.emotions, which_set, batch_size, parameters
        )
        self.val_data = self.data_reader.get_emotion_data(
            self.emotions, Set.VAL, batch_size, parameters
        )
        if weighted:
            self.class_weights = self.get_class_weights(which_set, parameters)
        else:
            self.class_weights = None

    @staticmethod
    def compute_mfccs(audio_tensor: tf.Tensor) -> tf.Tensor:

        # A 1024-point STFT with frames of 64 ms and 75% overlap.
        stfts = tf.signal.stft(
            audio_tensor, frame_length=1024, frame_step=256, fft_length=1024
        )
        spectrograms = tf.abs(stfts)

        # Warp the linear scale spectrograms into the mel-scale.
        num_spectrogram_bins = stfts.shape[-1]
        lower_edge_hertz, upper_edge_hertz, num_mel_bins = 80.0, 7600.0, 80
        linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins,
            num_spectrogram_bins,
            16000,
            lower_edge_hertz,
            upper_edge_hertz,
        )
        mel_spectrograms = tf.tensordot(
            spectrograms, linear_to_mel_weight_matrix, 1
        )
        mel_spectrograms.set_shape(
            spectrograms.shape[:-1].concatenate(
                linear_to_mel_weight_matrix.shape[-1:]
            )
        )

        # Compute a stabilized log to get log-magnitude mel-scale spectrograms.
        log_mel_spectrograms = tf.math.log(mel_spectrograms + 1e-6)

        # Compute MFCCs from log_mel_spectrograms and take the first 40.
        mfccs = tf.signal.mfccs_from_log_mel_spectrograms(
            log_mel_spectrograms
        )[..., :40]

        return mfccs

    @staticmethod
    def compute_spectrogram(audio_tensor: tf.Tensor) -> tf.Tensor:
        """
        Performs a short term fourier transform on the raw audio data to
        generate a spectrogram which can be used for classification.

        :param audio_tensor: The tensor containing raw audio data.
        :return:
        """
        stfts = tf.signal.stft(
            audio_tensor, frame_length=1024, frame_step=256, fft_length=1024
        )
        spectrograms = tf.abs(stfts)
        return spectrograms


if __name__ == "__main__":  # pragma: no cover
    from src.data.speech_data_reader import SpeechDataReader

    dr = SpeechDataReader()

    for speech, labels in dr.get_seven_emotion_data(Set.TEST):
        mfcc = SpeechEmotionClassifier.compute_mfccs(speech)
        print(mfcc.shape)

        spectrogram = SpeechEmotionClassifier.compute_spectrogram(speech)
        print(spectrogram.shape)
