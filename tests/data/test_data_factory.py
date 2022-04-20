"""Test the data factory class"""

import pytest
import tensorflow as tf

from src.data.data_factory import DataFactory, ImageDataReader, TextDataReader
from src.data.data_reader import Set


def test_data_reader_factory():
    text_reader = DataFactory.get_data_reader("text")
    assert isinstance(text_reader, TextDataReader)
    assert text_reader.name == "text"

    image_reader = DataFactory.get_data_reader("image")
    assert isinstance(image_reader, ImageDataReader)
    assert image_reader.name == "image"

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
