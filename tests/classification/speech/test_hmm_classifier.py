import glob
import os.path
import shutil

import numpy as np
import pytest

from src.classification.speech import HMMClassifier
from src.data.classwise_speech_data_reader import ClasswiseSpeechDataReader

CLASS_NAMES = [
    "angry",
    "surprise",
    "disgust",
    "happy",
    "fear",
    "sad",
    "neutral",
]


def test_hmm_initialization():
    classifier = HMMClassifier()
    assert not len(classifier.models.keys())
    assert not len(classifier.scaler.keys())
    assert not classifier.is_trained

    classifier.data_reader = ClasswiseSpeechDataReader(
        folder="tests/test_data/speech"
    )
    with pytest.raises(RuntimeError):
        # Error because not trained
        classifier.classify()


@pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
def test_hmm_workflow():
    files = glob.glob("tests/test_data/speech/train/**/*.wav")
    classifier = HMMClassifier()
    train_parameters = {"dataset": "meld"}
    classifier.data_reader = ClasswiseSpeechDataReader(
        folder="tests/test_data/speech"
    )

    try:
        for file in files:
            new_file = file[:-4] + "_copy.wav"
            new_file2 = file[:-4] + "_2copy.wav"
            new_file3 = file[:-4] + "_3copy.wav"
            shutil.copyfile(file, new_file)
            shutil.copyfile(file, new_file2)
            shutil.copyfile(file, new_file3)
        classifier.train(train_parameters)
        for file in files:
            new_file = file[:-4] + "_copy.wav"
            new_file2 = file[:-4] + "_2copy.wav"
            new_file3 = file[:-4] + "_3copy.wav"
            os.remove(new_file)
            os.remove(new_file2)
            os.remove(new_file3)
    except BaseException as e:
        files = glob.glob("tests/test_data/speech/train/**/*copy.wav")
        for file in files:
            os.remove(file)
        raise e

    shutil.rmtree("tests/temp/hmm", ignore_errors=True)
    save_parameters = {"save_path": "tests/temp/hmm"}
    classifier.save(save_parameters)
    for emotion in CLASS_NAMES:
        assert os.path.exists(f"tests/temp/hmm/{emotion}.pkl")
        assert os.path.exists(f"tests/temp/hmm/{emotion}_scaler.pkl")
    results = classifier.classify({"shuffle": False, "dataset": "meld"})
    assert isinstance(results, np.ndarray)
    assert results.shape == (7,)

    new_classifier = HMMClassifier()
    new_classifier.load(save_parameters)
    new_classifier.data_reader = ClasswiseSpeechDataReader(
        folder="tests/test_data/speech"
    )
    new_results = new_classifier.classify(
        {"shuffle": False, "dataset": "meld"}
    )
    assert np.array_equal(new_results, results)

    with pytest.raises(RuntimeError):
        new_classifier.save({"save_path": "tests/temp/hmm"})

    shutil.rmtree("tests/temp", ignore_errors=True)