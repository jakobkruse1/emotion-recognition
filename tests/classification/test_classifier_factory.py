"""Test the factory for classifiers"""
import pytest

from src.classification.classifier_factory import ClassifierFactory
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


def test_text_factory():
    classifier = ClassifierFactory.get("text", "nrclex", {})
    assert isinstance(classifier, NRCLexTextClassifier)

    classifier = ClassifierFactory.get("text", "bert", {"model_name": "123"})
    assert isinstance(classifier, BertClassifier)
    assert classifier.model_name == "123"

    classifier = ClassifierFactory.get("text", "distilbert", {})
    assert isinstance(classifier, DistilBertClassifier)

    with pytest.raises(ValueError):
        _ = ClassifierFactory.get("wrong", "bert", {})

    with pytest.raises(ValueError):
        _ = ClassifierFactory.get("text", "wrong", {})


def test_image_factory():
    classifier = ClassifierFactory.get("image", "vgg16", {})
    assert isinstance(classifier, VGG16Classifier)

    classifier = ClassifierFactory.get(
        "image", "cross_attention", {"model_name": "123"}
    )
    assert isinstance(classifier, CrossAttentionNetworkClassifier)
    assert classifier.parameters["model_name"] == "123"

    classifier = ClassifierFactory.get("image", "efficientnet", {})
    assert isinstance(classifier, MultiTaskEfficientNetB2Classifier)

    with pytest.raises(ValueError):
        _ = ClassifierFactory.get("wrong", "efficientnet", {})

    with pytest.raises(ValueError):
        _ = ClassifierFactory.get("image", "wrong", {})


def test_speech_factory():
    classifier = ClassifierFactory.get("speech", "mfcc_lstm", {})
    assert isinstance(classifier, MFCCLSTMClassifier)

    classifier = ClassifierFactory.get(
        "speech", "hubert", {"model_name": "123"}
    )
    assert isinstance(classifier, HuBERTClassifier)
    assert classifier.parameters["model_name"] == "123"

    classifier = ClassifierFactory.get("speech", "wav2vec2", {})
    assert isinstance(classifier, Wav2Vec2Classifier)

    classifier = ClassifierFactory.get("speech", "hmm", {})
    assert isinstance(classifier, HMMClassifier)

    classifier = ClassifierFactory.get("speech", "gmm", {})
    assert isinstance(classifier, GMMClassifier)

    classifier = ClassifierFactory.get("speech", "svm", {})
    assert isinstance(classifier, SVMClassifier)

    classifier = ClassifierFactory.get("speech", "byols", {})
    assert isinstance(classifier, BYOLSClassifier)

    with pytest.raises(ValueError):
        _ = ClassifierFactory.get("wrong", "efficientnet", {})

    with pytest.raises(ValueError):
        _ = ClassifierFactory.get("speech", "wrong", {})


def test_plant_factory():
    classifier = ClassifierFactory.get("plant", "plant_dense", {})
    assert isinstance(classifier, PlantDenseClassifier)

    classifier = ClassifierFactory.get(
        "plant", "plant_dense", {"model_name": "123"}
    )
    assert isinstance(classifier, PlantDenseClassifier)
    assert classifier.parameters["model_name"] == "123"

    classifier = ClassifierFactory.get("plant", "plant_lstm", {})
    assert isinstance(classifier, PlantLSTMClassifier)

    classifier = ClassifierFactory.get("plant", "plant_mfcc_cnn", {})
    assert isinstance(classifier, PlantMFCCCNNClassifier)

    classifier = ClassifierFactory.get("plant", "plant_mfcc_resnet", {})
    assert isinstance(classifier, PlantMFCCResnetClassifier)

    with pytest.raises(ValueError):
        _ = ClassifierFactory.get("wrong", "plant_mfcc_cnn", {})

    with pytest.raises(ValueError):
        _ = ClassifierFactory.get("plant", "wrong", {})


def test_watch_factory():
    classifier = ClassifierFactory.get("watch", "watch_dense", {})
    assert isinstance(classifier, WatchDenseClassifier)

    classifier = ClassifierFactory.get(
        "watch", "watch_dense", {"model_name": "123"}
    )
    assert isinstance(classifier, WatchDenseClassifier)
    assert classifier.parameters["model_name"] == "123"

    classifier = ClassifierFactory.get("watch", "watch_lstm", {})
    assert isinstance(classifier, WatchLSTMClassifier)

    classifier = ClassifierFactory.get("watch", "watch_random_forest", {})
    assert isinstance(classifier, WatchRandomForestClassifier)

    classifier = ClassifierFactory.get("watch", "watch_transformer", {})
    assert isinstance(classifier, WatchTransformerClassifier)

    classifier = ClassifierFactory.get("watch", "watch_xgboost", {})
    assert isinstance(classifier, WatchXGBoostClassifier)

    with pytest.raises(ValueError):
        _ = ClassifierFactory.get("wrong", "watch_xgboost", {})

    with pytest.raises(ValueError):
        _ = ClassifierFactory.get("watch", "wrong", {})
