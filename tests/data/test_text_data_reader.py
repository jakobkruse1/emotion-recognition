"""Test the text data reader"""
import os.path

from src.data.text_data_reader import Set, TextDataReader


def test_initialization():
    dr = TextDataReader()
    assert dr.name == "text"
    assert dr.folder == "data/train/text"
    for set_type in [Set.TRAIN, Set.VAL, Set.TEST]:
        file = os.path.join(dr.folder, dr.file_map[set_type])
        assert os.path.exists(file)
