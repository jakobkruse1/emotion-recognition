"""Test the factory for classifiers"""
import pytest

from src.classification.classifier_factory import ClassifierFactory
from src.classification.image import (
    CrossAttentionNetworkClassifier,
    MultiTaskEfficientNetB2Classifier,
    VGG16Classifier,
)
from src.classification.text import (
    BertClassifier,
    DistilBertClassifier,
    NRCLexTextClassifier,
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
