import numpy as np
import pytest

from src.emotion_set import (
    EkmanEmotions,
    EkmanNeutralEmotions,
    EmotionMapper,
    EmotionSetFactory,
    ThreeEmotions,
)


def test_emotion_factory():
    emotion_set = EmotionSetFactory.generate("three")
    assert isinstance(emotion_set, ThreeEmotions)
    emotion_set = EmotionSetFactory.generate("ekman")
    assert isinstance(emotion_set, EkmanEmotions)
    emotion_set = EmotionSetFactory.generate("neutral_ekman")
    assert isinstance(emotion_set, EkmanNeutralEmotions)
    with pytest.raises(ValueError):
        EmotionSetFactory.generate("wrong_name")


@pytest.mark.parametrize(
    "set_name,count,emotions",
    [
        ("three", 3, ["positive", "neutral", "negative"]),
        (
            "ekman",
            6,
            ["anger", "surprise", "disgust", "joy", "fear", "sadness"],
        ),
        (
            "neutral_ekman",
            7,
            [
                "anger",
                "surprise",
                "disgust",
                "joy",
                "fear",
                "sadness",
                "neutral",
            ],
        ),
    ],
)
def test_emotion_sets(set_name, count, emotions):
    emotion_set = EmotionSetFactory.generate(set_name)
    assert emotion_set.name == set_name
    assert emotion_set.emotion_count == count
    assert np.array(emotions == emotion_set.emotion_names).all()

    with pytest.raises(AssertionError):
        emotion_set.get_emotions(-1)
    with pytest.raises(AssertionError):
        test_emotions = np.zeros((5, 5))
        test_emotions[3, 3] = count
        emotion_set.get_emotions(test_emotions)

    test_emotions = np.reshape(list(range(2, 11)), (3, 3))
    test_emotions = test_emotions % count
    retrieved_emotions = emotion_set.get_emotions(test_emotions)
    for x, val in enumerate(retrieved_emotions):
        for y, emotion_val in enumerate(val):
            assert emotion_val == emotions[test_emotions[x, y]]
            assert emotion_val == emotion_set.get_emotions(test_emotions[x, y])


def test_emotion_mapper():
    mapper = EmotionMapper()
    for emotion in ["anger", "annoyance", "disapproval"]:
        assert mapper.map_emotion(emotion) == "anger"
    for emotion in ["disgust"]:
        assert mapper.map_emotion(emotion) == "disgust"
    for emotion in ["fear", "nervousness"]:
        assert mapper.map_emotion(emotion) == "fear"
    for emotion in [
        "joy",
        "amusement",
        "approval",
        "excitement",
        "gratitude",
        "love",
        "optimism",
        "relief",
        "pride",
        "admiration",
        "desire",
        "caring",
    ]:
        assert mapper.map_emotion(emotion) == "joy"
    for emotion in [
        "sadness",
        "disappointment",
        "embarrassment",
        "grief",
        "remorse",
    ]:
        assert mapper.map_emotion(emotion) == "sadness"
    for emotion in ["surprise", "realization", "confusion", "curiosity"]:
        assert mapper.map_emotion(emotion) == "surprise"
    for emotion in ["neutral"]:
        assert mapper.map_emotion(emotion) == "neutral"

    with pytest.raises(KeyError):
        _ = mapper.map_emotion("not_an_emotion")
