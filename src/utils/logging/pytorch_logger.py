""" Implement a logger for tensorflow keras models. """
from typing import Any, Dict

from torch import Tensor

from src.utils.logging.base_logger import BaseLogger


class TorchLogger(BaseLogger):
    """
    Implements a logger for pytorch models that receives relevant values every
    epoch and creates the logs from them.
    """

    def __init__(self):
        """
        Initialization method.
        """
        super().__init__()

    def log_epoch(self, data: Dict[str, Any]) -> None:
        """
        Epoch logging method. Pass a loss, val_loss, acc and val_acc
        every epoch.

        :param data: Dictionary with data to log.
        """
        for key in ["train_loss", "val_loss", "train_acc", "val_acc"]:
            item = (
                data[key].item()
                if isinstance(data[key], Tensor)
                else float(data[key])
            )
            self.logs[key].append(item)

    def log_end(self, data: Dict[str, Any]) -> None:
        """
        Log additional data at the end of the training if necessary.

        :param data: Data dictionary containing important logs.
        """
        self.logs.update(data)

    def log_start(self, data: Dict[str, Any]) -> None:
        """
        Log arbitrary data like training parameters etc.

        :param data: Data dictionary containing important data to log.
        """
        self.logs.update(data)
