"""Test the NRCLex Classifier"""

from src.classification.text import NRCLexTextClassifier


def test_initialization():
    classifier = NRCLexTextClassifier()

    classifier.train()
    classifier.save()
    classifier.load()


def test_classification():
    classifier = NRCLexTextClassifier()

    classifier.classify()
