"""Test the data factory class"""

import pytest
import tensorflow as tf

from src.data.data_factory import (
    BalancedImageDataReader,
    BalancedPlantExperimentDataReader,
    ComparisonImageDataReader,
    ComparisonSpeechDataReader,
    ComparisonTextDataReader,
    DataFactory,
    ImageDataReader,
    PlantExperimentDataReader,
    SpeechDataReader,
    TextDataReader,
    WatchExperimentDataReader,
)
from src.data.data_reader import Set


def test_data_reader_factory():
    text_reader = DataFactory.get_data_reader("text")
    assert isinstance(text_reader, TextDataReader)
    assert text_reader.name == "text"

    image_reader = DataFactory.get_data_reader("image")
    assert isinstance(image_reader, ImageDataReader)
    assert image_reader.name == "image"

    image_reader = DataFactory.get_data_reader("balanced_image")
    assert isinstance(image_reader, BalancedImageDataReader)
    assert image_reader.name == "balanced_image"

    speech_reader = DataFactory.get_data_reader("speech")
    assert isinstance(speech_reader, SpeechDataReader)
    assert speech_reader.name == "speech"

    plant_reader = DataFactory.get_data_reader("plant")
    assert isinstance(plant_reader, PlantExperimentDataReader)
    assert plant_reader.name == "plant"

    plant_reader = DataFactory.get_data_reader("balanced_plant")
    assert isinstance(plant_reader, BalancedPlantExperimentDataReader)
    assert plant_reader.name == "balanced_plant"

    watch_reader = DataFactory.get_data_reader("watch")
    assert isinstance(watch_reader, WatchExperimentDataReader)
    assert watch_reader.name == "watch"

    watch_reader = DataFactory.get_data_reader("comparison_text")
    assert isinstance(watch_reader, ComparisonTextDataReader)
    assert watch_reader.name == "comparison_text"

    watch_reader = DataFactory.get_data_reader("comparison_image")
    assert isinstance(watch_reader, ComparisonImageDataReader)
    assert watch_reader.name == "comparison_image"

    watch_reader = DataFactory.get_data_reader("comparison_speech")
    assert isinstance(watch_reader, ComparisonSpeechDataReader)
    assert watch_reader.name == "comparison_speech"

    with pytest.raises(ValueError):
        _ = DataFactory.get_data_reader("wrong")


def test_dataset_factory():
    for set_type in [Set.TRAIN, Set.VAL, Set.TEST]:
        text_data = DataFactory.get_dataset(
            "text", set_type, data_folder="tests/test_data/text"
        )
        assert isinstance(text_data, tf.data.Dataset)

    for set_type in [Set.TRAIN, Set.VAL, Set.TEST]:
        text_data = DataFactory.get_dataset(
            "text",
            set_type,
            emotions="three",
            data_folder="tests/test_data/text",
        )
        assert isinstance(text_data, tf.data.Dataset)

    with pytest.raises(ValueError):
        _ = DataFactory.get_dataset("wrong", Set.TEST)

    with pytest.raises(ValueError):
        _ = DataFactory.get_dataset(
            "text",
            Set.TEST,
            emotions="wrong",
            data_folder="tests/test_data/text",
        )
