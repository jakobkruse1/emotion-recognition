""" This file contains the BYOL-S speech emotion classifier """
import os
from typing import Dict

import numpy as np
import serab_byols
import tensorflow as tf
import torch
from alive_progress import alive_bar
from torch import nn

from src.classification.speech.speech_emotion_classifier import (
    SpeechEmotionClassifier,
)
from src.data.data_reader import Set


class BYOLSModel(nn.Module):
    """
    Pytorch model for the BYOL-S classifier models
    """

    def __init__(
        self, model_name: str, device: torch.device, parameters: Dict = None
    ) -> None:
        """
        Constructor for the model class that initializes the layers.

        :param model_name: Model name for the model to use
        :param device: Torch device to run the model on
        :param parameters: Dictionary with config parameters
        """
        super().__init__()
        parameters = parameters or {}
        self.model_name = model_name
        self.cfg_path = "models/speech/serab-byols/serab_byols/config.yaml"
        freeze = parameters.get("freeze", True)
        hidden = parameters.get("hidden", 1024)
        self.device = device
        self.checkpoints = {
            "cvt": "models/speech/serab-byols/checkpoints/"
            "cvt_s1-d1-e64_s2-d1-e256_s3-d1-e512_BYOLAs64x96-"
            "osandbyolaloss6373-e100-bs256-lr0003-rs42.pth",
            "default": "models/speech/serab-byols/checkpoints/"
            "default2048_BYOLAs64x96-2105311814-"
            "e100-bs256-lr0003-rs42.pth",
            "resnetish34": "models/speech/serab-byols/checkpoints/"
            "resnetish34_BYOLAs64x96-2105271915-e100-bs256"
            "-lr0003-rs42.pth",
        }

        self.model = serab_byols.load_model(
            self.checkpoints[model_name], model_name, cfg_path=self.cfg_path
        )

        if freeze:
            self.model.requires_grad = False

        self.hidden = nn.Linear(2048, hidden)
        self.relu = nn.ReLU()
        self.classifier = nn.Linear(hidden, 7)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model used for inference and training.

        :param x: A batch of raw speech data in a torch tensor
        :return: Tensor with the model output
        """
        embeddings = serab_byols.get_scene_embeddings(
            x, self.model, cfg_path=self.cfg_path
        )
        out = self.hidden(embeddings)
        out = self.relu(out)
        out = self.classifier(out)
        out = self.softmax(out)
        return out


class BYOLSClassifier(SpeechEmotionClassifier):
    """
    Class that implements a speech emotion classifier that uses BYOL-S features
    from https://github.com/GasserElbanna/serab-byols .
    """

    def __init__(self, parameters: Dict = None):
        """
        Initialize the BYOL-S emotion classifier

        :param name: The name for the classifier
        :param parameters: Some configuration parameters for the classifier
        """
        super().__init__("byols", parameters)
        parameters = parameters or {}
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )
        self.model = None

        from functools import partialmethod

        from tqdm import tqdm

        tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)

    def initialize_model(self, parameters: Dict) -> None:
        """
        Initializes a new and pretrained version of the BYOL-S model
        """
        model_name = parameters.get("model_name", "cvt")
        assert model_name in ["cvt", "resnetish34", "default"]
        self.model = BYOLSModel(model_name, self.device, parameters)

    def train(self, parameters: Dict = None, **kwargs) -> None:
        """
        Training method for MFCC-LSTM model

        :param parameters: Parameter dictionary used for training
        :param kwargs: Additional kwargs parameters
        """
        parameters = self.init_parameters(parameters, **kwargs)
        epochs = parameters.get("epochs", 20)
        learning_rate = parameters.get("learning_rate", 5e-5)
        batch_size = parameters.get("batch_size", 64)
        parameters["batch_size"] = batch_size
        patience = parameters.get("patience", 5)
        if "gpu" in parameters:  # pragma: no cover
            self.device = torch.device(
                f"cuda:{parameters['gpu']}"
                if torch.cuda.is_available()
                else "cpu"
            )

        if not self.model:
            self.initialize_model(parameters)
        self.model.to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=learning_rate
        )
        self.loss = torch.nn.CrossEntropyLoss().to(self.device)

        with tf.device("/cpu:0"):
            self.prepare_data(parameters)
            total_train_images = self.data_reader.get_labels(
                Set.TRAIN, parameters
            ).shape[0]
            batches = int(np.ceil(total_train_images / batch_size))
            total_val_images = self.data_reader.get_labels(
                Set.VAL, parameters
            ).shape[0]
            val_batches = int(np.ceil(total_val_images / batch_size))
        best_acc = 0
        waiting_for_improve = 0

        for epoch in range(epochs):
            running_loss = 0.0
            correct_sum = 0
            self.model.train()

            with alive_bar(
                batches, title=f"Epoch {epoch}/{epochs}", force_tty=True
            ) as bar:
                for batch, (data, labels) in enumerate(self.train_data):
                    self.optimizer.zero_grad()

                    data = torch.tensor(data.numpy()).to(self.device)
                    labels = torch.tensor(labels.numpy()).to(self.device)
                    pred = self.model(data)
                    loss_val = self.loss(pred, labels)
                    loss_val.backward()
                    self.optimizer.step()
                    running_loss += loss_val
                    _, predicts = torch.max(pred, 1)
                    correct_num = torch.eq(
                        predicts, torch.max(labels, 1)[1]
                    ).sum()
                    correct_sum += correct_num
                    bar.text = (
                        f" --> Loss: {running_loss /(batch+1):.3f}, "
                        f"Acc {correct_sum/((batch+1)*batch_size):.3f}"
                    )
                    bar()
            acc = correct_sum / total_train_images
            running_loss = running_loss / batches
            print(f"Epoch {epoch}: Acc {acc:.3f}, Loss {running_loss:.3f}")

            with torch.no_grad():
                running_loss = 0.0
                bingo_cnt = 0
                self.model.eval()

                for batch, (data, labels) in enumerate(self.val_data):
                    data = torch.tensor(data.numpy()).to(self.device)
                    labels = torch.tensor(labels.numpy()).to(self.device)
                    pred = self.model(data)
                    loss_val = self.loss(pred, labels)
                    running_loss += loss_val
                    _, predicts = torch.max(pred, 1)
                    correct_num = torch.eq(
                        predicts, torch.max(labels, 1)[1]
                    ).sum()
                    correct_sum += correct_num
                    bingo_cnt += correct_num.sum().cpu()
                running_loss = running_loss / val_batches
                acc = bingo_cnt / total_val_images
                acc = np.around(acc.numpy(), 4)
                if acc < best_acc:
                    waiting_for_improve += 1  # pragma: no cover
                else:
                    waiting_for_improve = 0  # pragma: no cover
                best_acc = max(acc, best_acc)
                print(
                    f"Epoch {epoch}: Val Acc {acc:.3f}, "
                    f"Val Loss {running_loss:.3f}"
                )
                if waiting_for_improve > patience:
                    break  # pragma: no cover

        self.is_trained = True

    def load(self, parameters: Dict = None, **kwargs) -> None:
        """
        Loading method that loads a previously trained model from disk.

        :param parameters: Parameters required for loading the model
        :param kwargs: Additional kwargs parameters
        """
        parameters = self.init_parameters(parameters, **kwargs)
        save_path = parameters.get(
            "save_path", "models/speech/byols/byols.pth"
        )
        saved_data = torch.load(save_path, map_location=self.device)
        with open(
            os.path.join(os.path.dirname(save_path), "model.txt"), "r"
        ) as file:
            model_name = file.read()
        self.model = BYOLSModel(model_name, self.device, parameters)
        self.model.load_state_dict(saved_data["model_state_dict"])
        self.model.eval()

    def save(self, parameters: Dict = None, **kwargs) -> None:
        """
        Saving method that saves a previously trained model on disk.

        :param parameters: Parameters required for storing the model
        :param kwargs: Additional kwargs parameters
        """
        if not self.is_trained:
            raise RuntimeError(
                "Model needs to be trained in order to save it!"
            )
        parameters = self.init_parameters(parameters, **kwargs)
        save_path = parameters.get(
            "save_path", "models/speech/byols/byols.pth"
        )
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save({"model_state_dict": self.model.state_dict()}, save_path)
        with open(
            os.path.join(os.path.dirname(save_path), "model.txt"), "w"
        ) as file:
            file.write(self.model.model_name)

    def classify(self, parameters: Dict = None, **kwargs) -> np.array:
        """
        Classification method used to classify emotions from speech

        :param parameters: Parameter dictionary used for classification
        :param kwargs: Additional kwargs parameters
        :return: An array with predicted emotion indices
        """
        parameters = self.init_parameters(parameters, **kwargs)
        which_set = parameters.get("which_set", Set.TEST)
        batch_size = parameters.get("batch_size", 64)
        with tf.device("/cpu:0"):
            dataset = self.data_reader.get_emotion_data(
                self.emotions, which_set, batch_size, parameters
            )

        if not self.model:
            raise RuntimeError(
                "Please load or train the model before inference!"
            )

        self.model.to(self.device)
        self.model.eval()

        results = np.empty((0, 7))
        with torch.no_grad():
            for data_batch, _ in dataset:
                data_batch = torch.tensor(data_batch.numpy()).to(self.device)
                out = self.model(data_batch)
                results = np.concatenate([results, out.cpu()], axis=0)

        return np.argmax(results, axis=1)


if __name__ == "__main__":  # pragma: no cover
    classifier = BYOLSClassifier()
    classifier.train(
        {
            "epochs": 1,
            "patience": 10,
            "learning_rate": 0.0003,
        }
    )
    classifier.save()
    classifier.load()
    emotions = classifier.classify()
    labels = classifier.data_reader.get_labels(Set.TEST)
    print(f"Labels Shape: {labels.shape}")
    print(f"Emotions Shape: {emotions.shape}")
    print(f"Accuracy: {np.sum(emotions == labels) / labels.shape[0]}")
