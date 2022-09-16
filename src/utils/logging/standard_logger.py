""" This file implements the base logger which is an interface for loggers. """

from typing import Any, Dict

from src.utils.logging.base_logger import BaseLogger


class StandardLogger(BaseLogger):
    """
    This class is a standard logger that logs random dictionaries.
    """

    def __init__(self) -> None:
        """
        Initialize the logger.
        """
        super().__init__()
        self.logs = {}

    def log_epoch(self, data: Dict[str, Any]) -> None:
        """
        Logging function that is called every epoch for logging epoch
        statistics. For example, it can be used to save the epoch's loss and
        metrics to the overall logging data.

        :param data: The dictionary containing epoch data and metrics.
        """
        for key, value in data.items():
            if key not in self.logs.keys():
                self.logs[key] = [value]
            else:
                self.logs[key].append(value)

    def log_end(self, data: Dict[str, Any]) -> None:
        """
        Logging function that is called after training for logging training
        statistics.

        :param data: The dictionary containing training data and metrics.
        """
        self.logs.update(data)

    def log_start(self, data: Dict[str, Any]) -> None:
        """
        Logging function that is called before training for logging training
        parameters and metrics. For example, it can be used to save
        initialization parameters or configuration details.

        :param data: The dictionary containing data and config.
        """
        self.logs.update(data)
