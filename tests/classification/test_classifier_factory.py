"""Test the factory for classifiers"""
import pytest

from src.classification.classifier_factory import ClassifierFactory
from src.classification.text import (
    BertClassifier,
    DistilBertClassifier,
    NRCLexTextClassifier,
)


def test_factory():
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
