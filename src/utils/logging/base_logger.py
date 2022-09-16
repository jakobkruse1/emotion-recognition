""" This file implements the base logger which is an interface for loggers. """

import json
import os
from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseLogger(ABC):
    """
    This class is the interface for all loggers.
    """

    def __init__(self) -> None:
        """
        Initialize the logger.
        """
        self.logs = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
        }

    @abstractmethod
    def log_epoch(self, data: Dict[str, Any]) -> None:
        """
        Logging function that is called every epoch for logging epoch
        statistics. For example, it can be used to save the epoch's loss and
        metrics to the overall logging data.

        :param data: The dictionary containing epoch data and metrics.
        """
        raise NotImplementedError("Abstract method!")  # pragma: no cover

    @abstractmethod
    def log_end(self, data: Dict[str, Any]) -> None:
        """
        Logging function that is called after training for logging training
        statistics. For example, it can be used to parse tensorflows history
        object after it is returned after training.

        :param data: The dictionary containing training data and metrics.
        """
        raise NotImplementedError("Abstract method!")  # pragma: no cover

    @abstractmethod
    def log_start(self, data: Dict[str, Any]) -> None:
        """
        Logging function that is called before training for logging training
        parameters and metrics. For example, it can be used to save
        initialization parameters or configuration details.

        :param data: The dictionary containing data and config.
        """
        raise NotImplementedError("Abstract method!")  # pragma: no cover

    def save_logs(self, folder: str) -> None:
        """
        This function saves the gathered data in a json file for review and
        plotting the statistics later.

        :param folder: The folder to store the statistics in.
        """
        if not os.path.isdir(folder):
            os.makedirs(folder, exist_ok=True)

        with open(os.path.join(folder, "statistics.json"), "w") as json_file:
            json.dump(self.logs, json_file)
