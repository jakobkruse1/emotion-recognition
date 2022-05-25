"""Package for speech based emotion classifiers"""

from .hubert_classifier import HuBERTClassifier  # noqa: F401
from .mfcc_lstm_classifier import MFCCLSTMClassifier  # noqa: F401
from .speech_emotion_classifier import SpeechEmotionClassifier  # noqa: F401
from .wav2vec2_classifier import Wav2Vec2Classifier  # noqa: F401
