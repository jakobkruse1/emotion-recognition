""" This file contains the EfficientNet facial emotion classifier """

import sys
from typing import Dict

import numpy as np
import pandas as pd
import torch
import torch.nn.init as init
import torch.utils.data as data
from alive_progress import alive_bar
from torch import nn
from torch.nn import functional as F
from torchvision import datasets, models
from tqdm import tqdm

from src.classification.image.image_emotion_classifier import (
    ImageEmotionClassifier,
)
from src.data.data_reader import Set


class DAN(nn.Module):
    def __init__(self, num_class=7, num_head=4, pretrained=True):
        super(DAN, self).__init__()

        resnet = models.resnet18(pretrained=pretrained)

        self.features = nn.Sequential(*list(resnet.children())[:-2])
        self.num_head = num_head
        for i in range(num_head):
            setattr(self, "cat_head%d" % i, CrossAttentionHead())
        self.sig = nn.Sigmoid()
        self.fc = nn.Linear(512, num_class)
        self.bn = nn.BatchNorm1d(num_class)

    def forward(self, x):
        x = self.features(x)
        heads = []
        for i in range(self.num_head):
            heads.append(getattr(self, "cat_head%d" % i)(x))

        heads = torch.stack(heads).permute([1, 0, 2])
        if heads.size(1) > 1:
            heads = F.log_softmax(heads, dim=1)

        out = self.fc(heads.sum(dim=1))
        out = self.bn(out)

        return out, x, heads


class CrossAttentionHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.sa = SpatialAttention()
        self.ca = ChannelAttention()
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        sa = self.sa(x)
        ca = self.ca(sa)

        return ca


class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1),
            nn.BatchNorm2d(256),
        )
        self.conv_3x3 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
        )
        self.conv_1x3 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=(1, 3), padding=(0, 1)),
            nn.BatchNorm2d(512),
        )
        self.conv_3x1 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(512),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        y = self.conv1x1(x)
        y = self.relu(self.conv_3x3(y) + self.conv_1x3(y) + self.conv_3x1(y))
        y = y.sum(dim=1, keepdim=True)
        out = x * y

        return out


class ChannelAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.attention = nn.Sequential(
            nn.Linear(512, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 512),
            nn.Sigmoid(),
        )

    def forward(self, sa):
        sa = self.gap(sa)
        sa = sa.view(sa.size(0), -1)
        y = self.attention(sa)
        out = sa * y

        return out


class AffinityLoss(nn.Module):
    def __init__(self, device, num_class=8, feat_dim=512):
        super(AffinityLoss, self).__init__()
        self.num_class = num_class
        self.feat_dim = feat_dim
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.device = device

        self.centers = nn.Parameter(
            torch.randn(self.num_class, self.feat_dim).to(device)
        )

    def forward(self, x, labels):
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


class PartitionLoss(nn.Module):
    def __init__(
        self,
    ):
        super(PartitionLoss, self).__init__()

    def forward(self, x):
        num_head = x.size(1)

        if num_head > 1:
            var = x.var(dim=1).mean()
            loss = torch.log(1 + num_head / (var + sys.float_info.epsilon))
        else:
            loss = 0

        return loss


class ImbalancedDatasetSampler(data.sampler.Sampler):
    def __init__(self, dataset, indices: list = None, num_samples: int = None):
        super().__init__(dataset)
        self.indices = (
            list(range(len(dataset))) if indices is None else indices
        )
        self.num_samples = (
            len(self.indices) if num_samples is None else num_samples
        )

        df = pd.DataFrame()
        df["label"] = self._get_labels(dataset)
        df.index = self.indices
        df = df.sort_index()

        label_to_count = df["label"].value_counts()

        weights = 1.0 / label_to_count[df["label"]]

        self.weights = torch.DoubleTensor(weights.to_list())

        # self.weights = self.weights.clamp(min=1e-5)

    def _get_labels(self, dataset):
        if isinstance(dataset, datasets.ImageFolder):
            return [x[1] for x in dataset.imgs]
        elif isinstance(dataset, torch.utils.data.Subset):
            return [dataset.dataset.imgs[i][1] for i in dataset.indices]
        else:
            raise NotImplementedError

    def __iter__(self):
        return (
            self.indices[i]
            for i in torch.multinomial(
                self.weights, self.num_samples, replacement=True
            )
        )

    def __len__(self):
        return self.num_samples


class CrossAttentionNetworkClassifier(ImageEmotionClassifier):
    """
    Class that implements an emotion classifier using a multi-head cross
    attention network based on ResNet50. Details can be found here:
    https://paperswithcode.com/paper/distract-your-attention-multi-head-cross
    """

    def __init__(self, parameters: Dict = None):
        """
        Initialize the CrossAttention emotion classifier

        :param name: The name for the classifier
        :param parameters: Some configuration parameters for the classifier
        """
        super().__init__("cross_attention", parameters)
        self.model = None
        self.optimizer = None
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )
        self.load({"save_path": "models/cache/cross_attention.pth"})

    def initialize_model(self, parameters: Dict) -> None:
        """
        Initializes a new and pretrained version of the CrossAttention model
        """
        self.model = DAN(num_class=7, num_head=4, pretrained=True)

    def train(self, parameters: Dict = None, **kwargs) -> None:
        """
        Training method for EfficientNet model

        :param parameters: Parameter dictionary used for training
        :param kwargs: Additional kwargs parameters
        """
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.enabled = True

        parameters = parameters or {}
        epochs = parameters.get("epochs", 20)
        which_set = parameters.get("which_set", Set.TRAIN)
        batch_size = parameters.get("batch_size", 64)
        learning_rate = parameters.get("learning_rate", 0.001)
        # patience = parameters.get("patience", 5)
        total_train_images = self.data_reader.get_labels(Set.TRAIN).shape[0]
        batches = int(np.ceil(total_train_images / batch_size))
        total_val_images = self.data_reader.get_labels(Set.VAL).shape[0]

        train_dataset = self.data_reader.get_emotion_data(
            self.emotions, which_set, batch_size
        )
        val_dataset = self.data_reader.get_emotion_data(
            self.emotions, Set.VAL, batch_size
        )

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
        for epoch in tqdm(range(epochs)):
            running_loss = 0.0
            correct_sum = 0
            iter_cnt = 0
            self.model.train()

            with alive_bar(
                batches, title=f"Epoch {epoch}", force_tty=True
            ) as bar:
                for batch, (imgs, targets) in enumerate(train_dataset):
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

            acc = correct_sum / total_train_images
            running_loss = running_loss / iter_cnt
            tqdm.write(
                f"[Epoch {epoch}] Training accuracy: {acc:.4f}. "
                f"Loss: {running_loss:.3f}]"
            )

            with torch.no_grad():
                running_loss = 0.0
                iter_cnt = 0
                bingo_cnt = 0
                self.model.eval()

                for data_batch, labels in val_dataset:
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
                best_acc = max(acc, best_acc)
                tqdm.write(
                    f"[Epoch {epoch}] Validation accuracy:{acc:.4f}. "
                    f"Loss:{running_loss:.3f}"
                )
                tqdm.write("best_acc:" + str(best_acc))

        self.is_trained = True

    def load(self, parameters: Dict = None, **kwargs) -> None:
        """
        Loading method that loads a previously trained model from disk.

        :param parameters: Parameters required for loading the model
        :param kwargs: Additional kwargs parameters
        """
        parameters = parameters or {}
        save_path = parameters.get(
            "save_path", "models/image/cross_attention/cross_attention.pth"
        )
        saved_data = torch.load(save_path, map_location=self.device)

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
        parameters = parameters or {}
        save_path = parameters.get(
            "save_path", "models/image/cross_attention/cross_attention.pth"
        )
        torch.save({"model_state_dict": self.model.state_dict()}, save_path)

    def classify(self, parameters: Dict = None, **kwargs) -> np.array:
        """
        Classification method used to classify emotions from images

        :param parameters: Parameter dictionary used for classification
        :param kwargs: Additional kwargs parameters
        :return: An array with predicted emotion indices
        """
        parameters = parameters or {}
        which_set = parameters.get("which_set", Set.TEST)
        batch_size = parameters.get("batch_size", 64)

        if not self.model:
            raise RuntimeError(
                "Please load or train the model before inference!"
            )
        self.model.to(self.device)
        self.model.eval()

        dataset = self.data_reader.get_seven_emotion_data(
            which_set, batch_size, shuffle=False
        )
        results = np.empty((0, 7))
        with torch.no_grad():
            for data_batch, labels in dataset:
                data_batch, labels = self.transform_data(data_batch, labels)
                out, feat, heads = self.model(data_batch)
                results = np.concatenate([results, out], axis=0)

        return np.argmax(results, axis=1)

    def transform_data(self, data, labels):
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


if __name__ == "__main__":  # pragma: no cover
    classifier = CrossAttentionNetworkClassifier()
    classifier.train()
    classifier.save()
    classifier.load()
    emotions = classifier.classify()
    labels = classifier.data_reader.get_labels(Set.TEST)
    print(f"Labels Shape: {labels.shape}")
    print(f"Emotions Shape: {emotions.shape}")
    print(f"Accuracy: {np.sum(emotions == labels) / labels.shape[0]}")
