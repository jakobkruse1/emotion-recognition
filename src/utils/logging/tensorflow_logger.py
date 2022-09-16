""" Implement a logger for tensorflow keras models. """
from typing import Any, Dict

from src.utils.logging.base_logger import BaseLogger


class KerasLogger(BaseLogger):
    """
    Implements a logger for keras models that uses tensorflow's
    history object and extracts the logs from it.
    """

    def __init__(self):
        """
        Initialization method.
        """
        super().__init__()

    def log_epoch(self, data: Dict[str, Any]) -> None:
        """
        Empty epoch logging method.
        Tensorflow logs all epoch logs itself and returns it in a history
        object. Therefore, no logging per epoch is required.

        :param data: Dictionary with data to log.
        """
        pass

    def log_end(self, data: Dict[str, Any]) -> None:
        """
        Log all the gathered training logs from tensorflow and keras.

        :param data: Data dictionary containing important logs.
            history: We expect a history object to be present here.
        """
        history = data["history"]
        training_logs = history.history
        print(training_logs)
        self.logs.update(
            {
                "train_loss": training_logs["loss"],
                "train_acc": training_logs["categorical_accuracy"],
                "val_loss": training_logs["val_loss"],
                "val_acc": training_logs["val_categorical_accuracy"],
            }
        )

    def log_start(self, data: Dict[str, Any]) -> None:
        """
        Log arbitrary data like training parameters etc.

        :param data: Data dictionary containing important data to log.
        """
        self.logs.update(data)
