""" This file contains the HuBERT speech emotion classifier """

import os
from typing import Dict

import numpy as np
import tensorflow as tf
import torch
from alive_progress import alive_bar
from torch import nn
from transformers import HubertConfig, HubertModel, Wav2Vec2Processor

from src.classification.speech.speech_emotion_classifier import (
    SpeechEmotionClassifier,
)
from src.data.data_reader import Set
from src.utils import logging, training_loop


class FinetuningHuBERTModel(nn.Module):
    """
    Pytorch model for the HuBERT classifier model
    """

    def __init__(self, device: torch.device, parameters: Dict = None) -> None:
        """
        Constructor for the model class that initializes the layers.

        :param parameters: Dictionary with config parameters
        """
        super().__init__()
        parameters = parameters or {}
        dropout = parameters.get("dropout", 0.1)
        freeze = parameters.get("freeze", False)
        extra_layer = parameters.get("extra_layer", None)

        model_config = HubertConfig(
            hidden_dropout=dropout,
            attention_dropout=dropout,
            num_hidden_layers=parameters.get("num_hidden_layers", 12),
        )
        cache_dir = "models/cache"
        self.device = device
        self.processor = Wav2Vec2Processor.from_pretrained(
            "facebook/hubert-large-ls960-ft", cache_dir=cache_dir
        )
        self.model = HubertModel.from_pretrained(
            "facebook/hubert-base-ls960",
            config=model_config,
            cache_dir=cache_dir,
        )
        if freeze:
            self.processor.requires_grad = False
            self.model.requires_grad = False
        if extra_layer:
            self.hidden = nn.Linear(114432, extra_layer)
            self.classifier = nn.Linear(extra_layer, 7)
        else:
            self.hidden = None
            self.classifier = nn.Linear(114432, 7)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model used for inference and training.

        :param x: A batch of raw speech data in a torch tensor
        :return: Tensor with the model output
        """
        input_values = self.processor(
            x, return_tensors="pt", sampling_rate=16000
        ).input_values[0]
        input_values = input_values.to(self.device)
        logits = self.model(input_values).last_hidden_state
        out = torch.flatten(logits, start_dim=1)
        if self.hidden:
            out = self.hidden(out)
        out = self.classifier(out)
        out = self.softmax(out)
        return out


class HuBERTClassifier(SpeechEmotionClassifier):
    """
    Class that implements a speech emotion classifier that uses
    the HuBERT model.
    """

    def __init__(self, parameters: Dict = None) -> None:
        """
        Initialize the HuBERT emotion classifier

        :param parameters: Some configuration parameters for the classifier
        """
        super().__init__("hubert", parameters)
        tf.get_logger().setLevel("ERROR")
        self.model = None
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )
        self.logger = logging.TorchLogger()
        self.logger.log_start({"init_parameters": parameters})

    def initialize_model(self, parameters: Dict) -> None:
        """
        Initializes a new and pretrained version of the HuBERT model

        :param parameters: Parameters for the model initialization.
        """
        self.model = FinetuningHuBERTModel(self.device, parameters)

    def train(self, parameters: Dict = None, **kwargs) -> None:
        """
        Training method for HuBERT model

        :param parameters: Parameter dictionary used for training
        :param kwargs: Additional kwargs parameters
        """
        parameters = self.init_parameters(parameters, **kwargs)
        self.logger.log_start({"train_parameters": parameters})
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
            train_acc = correct_sum / total_train_images
            train_loss = running_loss / batches
            print(f"Epoch {epoch}: Acc {train_acc:.3f}, Loss {train_loss:.3f}")

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
                self.logger.log_epoch(
                    {
                        "train_acc": train_acc,
                        "train_loss": train_loss,
                        "val_acc": acc,
                        "val_loss": running_loss,
                    }
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
            "save_path", "models/speech/hubert/hubert.pth"
        )
        saved_data = torch.load(save_path, map_location=self.device)
        self.model = FinetuningHuBERTModel(self.device, parameters)
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
            "save_path", "models/speech/hubert/hubert.pth"
        )
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save({"model_state_dict": self.model.state_dict()}, save_path)
        self.logger.save_logs(os.path.dirname(save_path))

    def classify(self, parameters: Dict = None, **kwargs) -> np.array:
        """
        Classification method used to classify emotions from images

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


def _main():  # pragma: no cover
    classifier = HuBERTClassifier()
    parameters = {
        "epochs": 50,
        "patience": 10,
        "learning_rate": 5e-05,
        "dropout": 0.1,
        "num_hidden_layers": 10,
        "freeze": False,
        "extra_layer": 0,
        "batch_size": 64,
    }
    save_path = "models/speech/hubert"
    training_loop(classifier, parameters, save_path)


if __name__ == "__main__":  # pragma: no cover
    _main()
