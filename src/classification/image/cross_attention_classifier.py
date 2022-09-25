""" This file contains the CrossAttention facial emotion classifier. """
import os
import sys
from typing import Dict, Tuple

import numpy as np
import tensorflow as tf
import torch
from alive_progress import alive_bar
from torchvision import models
from tqdm import tqdm

from src.classification.image.image_emotion_classifier import (
    ImageEmotionClassifier,
)
from src.data.data_reader import Set
from src.utils import logging, training_loop


class DAN(torch.nn.Module):
    """
    This class implements the "Distract Your Attention" (DAN) network,
    which was introduced in https://github.com/yaoing/DAN
    """

    def __init__(
        self, num_class: int = 7, num_head: int = 4, pretrained: bool = True
    ) -> None:
        """
        Initialize a new DAN network

        :param num_class: Number of classes to classify
        :param num_head: Number of heads for the network - paper suggests 4
        :param pretrained: Flag to decide using pretrained resnet or not
        """
        super(DAN, self).__init__()

        resnet = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT if pretrained else None
        )

        self.features = torch.nn.Sequential(*list(resnet.children())[:-2])
        self.num_head = num_head
        for i in range(num_head):
            setattr(self, "cat_head%d" % i, CrossAttentionHead())
        self.sig = torch.nn.Sigmoid()
        self.fc = torch.nn.Linear(512, num_class)
        self.bn = torch.nn.BatchNorm1d(num_class)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the DAN network.

        :param x: The tensor x to pass through the network;
        :return: Tuple containing three elements:
            0: Output of the classifier
            1: Resnet output features
            2: Output after the attention heads before the classifier
        """
        x = self.features(x)
        heads = []
        for i in range(self.num_head):
            heads.append(getattr(self, "cat_head%d" % i)(x))

        heads = torch.stack(heads).permute([1, 0, 2])
        if heads.size(1) > 1:
            heads = torch.nn.functional.log_softmax(heads, dim=1)

        out = self.fc(heads.sum(dim=1))
        out = self.bn(out)

        return out, x, heads


class CrossAttentionHead(torch.nn.Module):
    """
    Implementation of a CrossAttention Head in pytorch.
    Taken from the original paper without changes.
    """

    def __init__(self) -> None:
        """
        Initialization function for the CrossAttention head that creates
        the SpatialAttention and ChannelAttention layers.
        """
        super().__init__()
        self.sa = SpatialAttention()
        self.ca = ChannelAttention()
        self.init_weights()

    def init_weights(self) -> None:
        """
        Function that initializes the weights of all layers.
        """
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the AttentionHead layer.

        :param x: tensor to pass through the layer
        :return: Output of the layer
        """
        sa = self.sa(x)
        ca = self.ca(sa)

        return ca


class SpatialAttention(torch.nn.Module):
    """
    SpatialAttention layer that is a part of the CrossAttentionHead.
    """

    def __init__(self) -> None:
        """
        Initializer function that creates the SpatialAttention layer
        """
        super().__init__()
        self.conv1x1 = torch.nn.Sequential(
            torch.nn.Conv2d(512, 256, kernel_size=1),
            torch.nn.BatchNorm2d(256),
        )
        self.conv_3x3 = torch.nn.Sequential(
            torch.nn.Conv2d(256, 512, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(512),
        )
        self.conv_1x3 = torch.nn.Sequential(
            torch.nn.Conv2d(256, 512, kernel_size=(1, 3), padding=(0, 1)),
            torch.nn.BatchNorm2d(512),
        )
        self.conv_3x1 = torch.nn.Sequential(
            torch.nn.Conv2d(256, 512, kernel_size=(3, 1), padding=(1, 0)),
            torch.nn.BatchNorm2d(512),
        )
        self.relu = torch.nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the SpatialAttention layer

        :param x: The tensor to pass through the layer
        :return: The output tensor
        """
        y = self.conv1x1(x)
        y = self.relu(self.conv_3x3(y) + self.conv_1x3(y) + self.conv_3x1(y))
        y = y.sum(dim=1, keepdim=True)
        out = x * y

        return out


class ChannelAttention(torch.nn.Module):
    """
    The ChannelAttention layer, which is the second part of the
    CrossAttentionHead.
    """

    def __init__(self) -> None:
        """
        Initialization function
        """
        super().__init__()
        self.gap = torch.nn.AdaptiveAvgPool2d(1)
        self.attention = torch.nn.Sequential(
            torch.nn.Linear(512, 32),
            torch.nn.BatchNorm1d(32),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(32, 512),
            torch.nn.Sigmoid(),
        )

    def forward(self, sa: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the ChannelAttention layer

        :param sa: Input into the layer
            (should be the output of SpatialAttention)
        :return: Output of the ChannelAttention layer
        """
        sa = self.gap(sa)
        sa = sa.view(sa.size(0), -1)
        y = self.attention(sa)
        out = sa * y

        return out


class AffinityLoss(torch.nn.Module):
    """
    Affinity Loss function that is supposed to increase
    the inter-class distances while decreasing the intra-class distances.
    For more details about the computation go to chapter 3.1.1 of the paper.
    """

    def __init__(
        self, device: torch.device, num_class: int = 7, feat_dim: int = 512
    ) -> None:
        """
        Initializer for the Affinity Loss function

        :param device: Torch device to use for the computation
        :param num_class: Number of classes to predict
        :param feat_dim: Dimensionality of the features output
        """
        super(AffinityLoss, self).__init__()
        self.num_class = num_class
        self.feat_dim = feat_dim
        self.gap = torch.nn.AdaptiveAvgPool2d(1)
        self.device = device

        self.centers = torch.nn.Parameter(
            torch.randn(self.num_class, self.feat_dim).to(device)
        )

    def forward(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the loss function.

        :param x: features of the DAN model. Output after resnet.
        :param labels: Labels for the inputs to classify.
        :return: Loss function value
        """
        x = self.gap(x).view(x.size(0), -1)

        batch_size = x.size(0)
        distmat = (
            torch.pow(x, 2)
            .sum(dim=1, keepdim=True)
            .expand(batch_size, self.num_class)
            + torch.pow(self.centers, 2)
            .sum(dim=1, keepdim=True)
            .expand(self.num_class, batch_size)
            .t()
        )
        distmat.addmm_(x, self.centers.t(), beta=1, alpha=-2)

        classes = torch.arange(self.num_class).long().to(self.device)
        mask = labels.eq(classes.expand(batch_size, self.num_class))

        dist = distmat * mask.float()
        dist = dist / self.centers.var(dim=0).sum()

        loss = dist.clamp(min=1e-12, max=1e12).sum() / batch_size

        return loss


class PartitionLoss(torch.nn.Module):
    """
    Partition loss function that maximizes the variance among attention maps.
    Refer to chapter 3.3.1 for more details.
    """

    def __init__(self) -> None:
        """
        Initializer function
        """
        super(PartitionLoss, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the loss function that computes the loss value

        :param x: The input tensor to compute the loss for. Should contain the
            values of the tensors after the DAN heads.
        :return: Loss value in a tensor.
        """
        num_head = x.size(1)

        if num_head > 1:
            var = x.var(dim=1).mean()
            loss = torch.log(1 + num_head / (var + sys.float_info.epsilon))
        else:
            loss = 0  # pragma: no cover

        return loss


class CrossAttentionNetworkClassifier(ImageEmotionClassifier):
    """
    Class that implements an emotion classifier using a multi-head cross
    attention network based on ResNet50. Details can be found here:
    https://paperswithcode.com/paper/distract-your-attention-multi-head-cross
    """

    def __init__(self, parameters: Dict = None):
        """
        Initialize the CrossAttention emotion classifier

        :param parameters: Some configuration parameters for the classifier
        """
        super().__init__("cross_attention", parameters)
        self.model = None
        self.optimizer = None
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )
        self.logger = logging.TorchLogger()

    def initialize_model(self, parameters: Dict) -> None:
        """
        Initializes a new and pretrained version of the CrossAttention model
        """
        self.model = DAN(num_class=7, num_head=4, pretrained=True)
        self.logger.log_start({"init_parameters": parameters})

    def train(self, parameters: Dict = None, **kwargs) -> None:
        """
        Training method for CrossAttention model.
        Taken from https://github.com/yaoing/DAN and adapted slightly

        :param parameters: Parameter dictionary used for training
        :param kwargs: Additional kwargs parameters
        """
        if torch.cuda.is_available():  # pragma: no cover
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.enabled = True

        parameters = self.init_parameters(parameters, **kwargs)
        self.logger.log_start({"train_parameters": parameters})
        epochs = parameters.get("epochs", 50)
        batch_size = parameters.get("batch_size", 64)
        learning_rate = parameters.get("learning_rate", 0.001)
        patience = parameters.get("patience", 10)
        if "gpu" in parameters:
            self.device = torch.device(
                f"cuda:{parameters['gpu']}"
                if torch.cuda.is_available()
                else "cpu"
            )

        with tf.device("/cpu:0"):
            total_train_images = self.data_reader.get_labels(Set.TRAIN).shape[
                0
            ]
            batches = int(np.ceil(total_train_images / batch_size))
            total_val_images = self.data_reader.get_labels(Set.VAL).shape[0]
            self.prepare_data(parameters)

        if not self.model:
            self.initialize_model(parameters)
        self.model.to(self.device)

        criterion_cls = torch.nn.CrossEntropyLoss().to(self.device)
        criterion_af = AffinityLoss(self.device, num_class=7)
        criterion_pt = PartitionLoss()

        params = list(self.model.parameters()) + list(
            criterion_af.parameters()
        )
        self.optimizer = torch.optim.Adam(
            params, learning_rate, weight_decay=0
        )
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.optimizer, gamma=0.6
        )

        best_acc = 0
        waiting_for_improve = 0
        for epoch in tqdm(range(epochs)):
            running_loss = 0.0
            correct_sum = 0
            iter_cnt = 0
            self.model.train()

            with alive_bar(
                batches, title=f"Epoch {epoch}", force_tty=True
            ) as bar:
                for batch, (imgs, targets) in enumerate(self.train_data):
                    imgs, targets = self.transform_data(imgs, targets)
                    iter_cnt += 1
                    self.optimizer.zero_grad()

                    imgs = imgs.to(self.device)
                    targets = targets.to(self.device)

                    out, feat, heads = self.model(imgs)

                    loss = (
                        criterion_cls(out, targets)
                        + criterion_af(feat, targets)
                        + criterion_pt(heads)
                    )

                    loss.backward()
                    self.optimizer.step()

                    running_loss += loss
                    _, predicts = torch.max(out, 1)
                    correct_num = torch.eq(
                        predicts, torch.max(targets, 1)[1]
                    ).sum()
                    correct_sum += correct_num
                    bar()

            train_acc = correct_sum / total_train_images
            train_loss = running_loss / iter_cnt
            tqdm.write(
                f"[Epoch {epoch}] Training accuracy: {train_acc:.4f}. "
                f"Loss: {train_loss:.3f}"
            )

            with torch.no_grad():
                running_loss = 0.0
                iter_cnt = 0
                bingo_cnt = 0
                self.model.eval()

                for data_batch, labels in self.val_data:
                    imgs, targets = self.transform_data(data_batch, labels)
                    out, feat, heads = self.model(imgs)

                    loss = (
                        criterion_cls(out, targets)
                        + criterion_af(feat, targets)
                        + criterion_pt(heads)
                    )

                    running_loss += loss
                    iter_cnt += 1
                    _, predicts = torch.max(out, 1)
                    correct_num = torch.eq(predicts, torch.max(targets, 1)[1])
                    bingo_cnt += correct_num.sum().cpu()

                running_loss = running_loss / iter_cnt
                scheduler.step()

                acc = bingo_cnt / total_val_images
                acc = np.around(acc.numpy(), 4)
                if acc < best_acc:
                    waiting_for_improve += 1  # pragma: no cover
                else:
                    waiting_for_improve = 0  # pragma: no cover
                best_acc = max(acc, best_acc)
                tqdm.write(
                    f"[Epoch {epoch}] Validation accuracy:{acc:.4f}. "
                    f"Loss:{running_loss:.3f}"
                )
                tqdm.write("best_acc:" + str(best_acc))
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
        save_path = parameters.get("save_path", "models/image/cross_attention")
        saved_data = torch.load(
            os.path.join(save_path, "cross_attention.pth"),
            map_location=self.device,
        )

        self.model = DAN(num_class=7, num_head=4, pretrained=False)
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
        save_path = parameters.get("save_path", "models/image/cross_attention")
        os.makedirs(save_path, exist_ok=True)
        torch.save(
            {"model_state_dict": self.model.state_dict()},
            os.path.join(save_path, "cross_attention.pth"),
        )
        self.logger.save_logs(save_path)

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

        if not self.model:
            raise RuntimeError(
                "Please load or train the model before inference!"
            )
        self.model.to(self.device)
        self.model.eval()
        with tf.device("/cpu:0"):
            dataset = self.data_reader.get_seven_emotion_data(
                which_set, batch_size, parameters
            )
        results = np.empty((0, 7))
        with torch.no_grad():
            for data_batch, labels in dataset:
                data_batch, labels = self.transform_data(data_batch, labels)
                out, feat, heads = self.model(data_batch)
                results = np.concatenate([results, out.cpu()], axis=0)

        return np.argmax(results, axis=1)

    def transform_data(
        self, data: tf.Tensor, labels: tf.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Transforming method that transforms the data to torch tensors
        in the correct format.

        :param data: Tensor containing a batch of images
        :param labels: Tensor containing a batch of labels
        :return: torch tensors with images and labels
        """
        data = data.numpy() / 255.0
        data = data - np.array([0.485, 0.456, 0.406]) / np.array(
            [0.229, 0.224, 0.225]
        )
        data = np.moveaxis(data, 3, 1)
        data = torch.tensor(data, dtype=torch.float32)
        labels = labels.numpy()
        data = data.to(self.device)
        labels = torch.tensor(labels, dtype=torch.float32)
        labels = labels.to(self.device)
        return data, labels


def _main():  # pragma: no cover
    classifier = CrossAttentionNetworkClassifier()
    parameters = {
        "learning_rate": 0.0003,
        "augment": False,
        "weighted": False,
        "balanced": False,
    }
    save_path = "models/image/cross_attention"
    training_loop(classifier, parameters, save_path)


if __name__ == "__main__":  # pragma: no cover
    _main()
