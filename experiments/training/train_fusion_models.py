""" This file trains different fusion models and compares them on the
experiment data. """

import itertools
import os
import sys

from src.classification.fusion import FusionClassifier
from src.data.data_reader import Set
from src.utils.metrics import accuracy, per_class_accuracy


def main():
    parameters = {
        "epochs": 500,
        "batch_size": 64,
        "patience": 25,
        "learning_rate": 0.003,
        "hidden_size": 64,
    }
    for use_mod in itertools.product([True, False], repeat=3):
        use_im, use_wa, use_pl = use_mod
        if not (use_im or use_wa or use_pl):
            continue
        items = 0
        modalities = []
        if use_im:
            modalities.append("image")
            items += 7
        if use_wa:
            modalities.append("watch")
            items += 7
        if use_pl:
            modalities.append("plant")
            items += 7
        this_params = parameters.copy()
        this_params["modalities"] = modalities
        this_params["input_elements"] = items
        mod_path = "".join(["i", "w", "p"][i] for i in range(3) if use_mod[i])
        save_path = f"models/fusion/fusion_{mod_path}"
        this_params["save_path"] = save_path
        classifier = FusionClassifier()
        if not os.path.exists(save_path) or "train" in sys.argv:
            classifier.train(parameters)
            classifier.save(parameters)
        classifier.load(parameters)
        emotions = classifier.classify(parameters)
        labels = classifier.data_reader.get_labels(Set.TEST, parameters)
        print(
            f"-----\nFusion with {'Image, ' if use_im else ''}"
            f"{'Watch, ' if use_wa else ''}{'Plant' if use_pl else ''}"
        )
        print(f"Accuracy: {accuracy(labels, emotions)}")
        print(f"Per Class Accuracy: {per_class_accuracy(labels, emotions)}")


if __name__ == "__main__":
    main()
