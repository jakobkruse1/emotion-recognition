"""
This package contains a few image emotion classifiers based on different
classifier architectures.
"""

from .cross_attention_classifier import (  # noqa: F401
    CrossAttentionNetworkClassifier,
)
from .efficientnet_classifier import (  # noqa: F401
    MultiTaskEfficientNetB2Classifier,
)
from .image_emotion_classifier import ImageEmotionClassifier  # noqa: F401
from .vgg16_classifier import VGG16Classifier  # noqa: F401
