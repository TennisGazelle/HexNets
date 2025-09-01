from abc import ABC, abstractmethod
import numpy as np
import pathlib

from src.networks.activation.activations import BaseActivation
from src.networks.loss.loss import BaseLoss
from src.networks.activation.Sigmoid import Sigmoid
from src.networks.loss.MeanSquaredErrorLoss import MeanSquaredErrorLoss


class BaseNeuralNetwork(ABC):
    def __init_subclass__(self, **kwargs):
        self.display_name = kwargs.get("display_name", self.__class__.__name__.lower())

    def __init__(
        self, learning_rate: float = 0.01, activation: BaseActivation = Sigmoid, loss: BaseLoss = MeanSquaredErrorLoss
    ):
        self.learning_rate = learning_rate
        self.activation = activation
        self.loss = loss

    def init_training_metrics(self):
        return {"loss": [], "accuracy": [], "r_squared": []}

    # @abstractmethod
    # def _init_from_file(self, filepath):
    #     pass

    @abstractmethod
    def show_stats(self):
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
    def graph_structure(self, detail="", output_dir: pathlib.Path = None):
        pass

    @abstractmethod
    def get_metrics_json(self):
        pass
