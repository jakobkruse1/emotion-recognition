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
    classifier.data_reader = TextDataReader(folder="tests/test_data/text")

    results = classifier.classify({"batch_size": 5, "which_set": Set.TRAIN})
    assert results.shape == (30,)

    # TODO: Might add sanity checks for results here not only check shape
