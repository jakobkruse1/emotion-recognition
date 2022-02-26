"""Test the NRCLex Classifier"""

from src.classification.text import NRCLexTextClassifier
from src.data.text_data_reader import Set, TextDataReader
from src.emotion_set import EkmanNeutralEmotions, EmotionMapper


def test_initialization():
    classifier = NRCLexTextClassifier()

    emotion_mapper = EmotionMapper()
    emotion_set = EkmanNeutralEmotions()
    for key, value in classifier.emotion_map.items():
        correct_emotion = emotion_mapper.map_emotion(key)
        classifier_emotion = emotion_set.get_emotions(value)
        assert correct_emotion == classifier_emotion

    # All do nothing here
    classifier.train()
    classifier.save()
    classifier.load()


def test_classification():
    classifier = NRCLexTextClassifier()
    classifier.data_reader = TextDataReader(folder="tests/test_data")
    classifier.data_reader.file_map[Set.TRAIN] = "text_test.csv"

    results = classifier.classify({"batch_size": 5, "set": Set.TRAIN})
    assert results.shape == (30,)
