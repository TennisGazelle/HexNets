from abc import ABC, abstractmethod
import numpy as np
import pathlib
from typing import Tuple
import matplotlib.pyplot as plt

from networks.activation.activations import BaseActivation
from networks.loss.loss import BaseLoss
from networks.learning_rate.learning_rate import BaseLearningRate
from networks.activation.Sigmoid import Sigmoid
from networks.loss.MeanSquaredErrorLoss import MeanSquaredErrorLoss
from networks.learning_rate.ConstantLearningRate import ConstantLearningRate
from networks.learning_rate.learning_rate import get_learning_rate
from networks.metrics import Metrics


class BaseNeuralNetwork(ABC):
    def __init_subclass__(self, **kwargs):
        self.display_name = kwargs.get("display_name", self.__class__.__name__.lower())

    def __init__(
        self, learning_rate = "constant", activation: BaseActivation = Sigmoid, loss: BaseLoss = MeanSquaredErrorLoss
    ):
        # learning_rate can be a string (function name) or BaseLearningRate instance
        if isinstance(learning_rate, str):
            # Default to 0.01 for constant learning rate
            self.learning_rate_fn = get_learning_rate(learning_rate, learning_rate=0.01)
        elif isinstance(learning_rate, BaseLearningRate):
            self.learning_rate_fn = learning_rate
        else:
            # Backward compatibility: if float, use constant with that value
            self.learning_rate_fn = ConstantLearningRate(learning_rate=learning_rate)

        self.activation = activation
        self.loss = loss
        self.training_metrics = Metrics()
        self.epochs_completed = 0
        self.data_iteration = 0  # Track current data element index

    # def init_training_metrics(self):
    #     return {"loss": [], "accuracy": [], "r_squared": []}

    # @abstractmethod
    # def _init_from_file(self, filepath):
    #     pass

    @abstractmethod
    def show_stats(self):
        pass

    @abstractmethod
    def show_latest_metrics(self):
        pass

    @abstractmethod
    def save(self, filepath) -> None:
        pass

    @abstractmethod
    def load(filepath) -> "BaseNeuralNetwork":
        pass

    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def backward(self, activations: np.ndarray, target: np.ndarray, apply_delta_W: bool = True):
        pass

    @abstractmethod
    def apply_delta_W(self):
        pass

    @abstractmethod
    def train(self, data, epochs: int = 1):
        pass

    @abstractmethod
    def test(self, x):
        pass

    @abstractmethod
    def graph_weights(self, activation_only=True, detail="", output_dir: pathlib.Path = None):
        pass

    @abstractmethod
    def graph_structure(self, detail="", output_dir: pathlib.Path = None) -> Tuple[str, plt.Figure]:
        pass

    @abstractmethod
    def get_metrics_json(self):
        pass
