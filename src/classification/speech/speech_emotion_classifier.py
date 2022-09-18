""" Base class for all speech emotion classifiers """

from abc import abstractmethod
from typing import Dict

import librosa
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
        """
        Function that computes MFCC features from an audio tensor.

        :param audio_tensor: The tensor containing raw audio data.
        :return: tensor of mfcc features.
        """

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
    def compute_spectrogram(
        audio_tensor: tf.Tensor,
    ) -> tf.Tensor:  # pragma: no cover
        """
        Performs a short term fourier transform on the raw audio data to
        generate a spectrogram which can be used for classification.

        :param audio_tensor: The tensor containing raw audio data.
        :return: tensor with spectrogram data.
        """
        stfts = tf.signal.stft(
            audio_tensor, frame_length=1024, frame_step=256, fft_length=1024
        )
        spectrograms = tf.abs(stfts)
        return spectrograms

    @staticmethod
    def get_mixed_features(
        data: np.ndarray, parameters: Dict = None
    ) -> np.ndarray:
        """
        Feature construction method for speech classifiers that use
        both mfcc and other audio features.

        :param data: The raw audio array of one sentence.
        :param parameters: Parameter dictionary
        :return: The features for the audio sentence.
        """
        sr = 16000
        frame_t = 0.025
        frame_n = round(sr * frame_t)
        hop_n = round(0.01 * sr)
        mfcc_num = parameters.get("mfcc_num", 40)
        ethresh = 0.01

        # RMSE per frame
        s, phase = librosa.magphase(
            librosa.stft(y=data, win_length=frame_n, hop_length=hop_n)
        )
        rmst = librosa.feature.rms(S=s, hop_length=hop_n)

        elocs = np.where(rmst > ethresh)[1]
        if elocs.size == 0:
            elocs = [60, 249]
        eloc = np.arange(elocs[0], elocs[-1] + 1)
        rms = rmst[:, eloc]

        # MFCC per frame
        mfcct = librosa.feature.mfcc(
            y=data, sr=sr, n_fft=frame_n, hop_length=hop_n, n_mfcc=mfcc_num
        )
        mfcc = mfcct[:, eloc]
        #         print('Shape of MFCC', MFCC.shape)

        _cent = librosa.feature.spectral_centroid(
            y=data, sr=sr, n_fft=frame_n, hop_length=hop_n
        )
        cent = _cent[:, eloc]

        _rolloff = librosa.feature.spectral_rolloff(
            y=data, sr=sr, n_fft=frame_n, hop_length=hop_n
        )
        rolloff = _rolloff[:, eloc]

        # Zero Crossing Rate per frame
        zcrt = librosa.feature.zero_crossing_rate(
            y=data, frame_length=frame_n, hop_length=hop_n
        )
        zcr = zcrt[:, eloc]

        mfcc = np.mean(mfcc, axis=1)
        zcr = np.mean(zcr, axis=1)
        rms = np.mean(rms, axis=1)
        cent = np.mean(cent, axis=1)
        rolloff = np.mean(rolloff, axis=1)
        mfcc = np.reshape(mfcc, (len(mfcc), 1))
        zcr = np.reshape(zcr, (len(zcr), 1))
        rms = np.reshape(rms, (len(rms), 1))
        cent = np.reshape(cent, (len(cent), 1))
        rolloff = np.reshape(rolloff, (len(rolloff), 1))

        features = np.vstack((mfcc, zcr, rms, cent, rolloff))
        return features


def _main():  # pragma: no cover
    from src.data.speech_data_reader import SpeechDataReader

    dr = SpeechDataReader()

    for speech, labels in dr.get_seven_emotion_data(Set.TEST):
        mfcc = SpeechEmotionClassifier.compute_mfccs(speech)
        print(mfcc.shape)

        spectrogram = SpeechEmotionClassifier.compute_spectrogram(speech)
        print(spectrogram.shape)


if __name__ == "__main__":  # pragma: no cover
    _main()
