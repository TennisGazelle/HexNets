from abc import ABC, abstractmethod
from src.networks.activations import BaseActivation
from src.networks.loss import BaseLoss

# === Base class ===
class BaseNeuralNetwork(ABC):
    def __init__(self, n: int, activation: BaseActivation, loss: BaseLoss):
        self.n = n
        self.activation = activation
        self.loss = loss

    @abstractmethod
    def save(self, filepath):
        pass

    @abstractmethod
    def load(self, filepath):
        pass

    @abstractmethod
    def train(self, data):
        pass

    @abstractmethod
    def test(self, x):
        pass

    def graph(self, detail=""):
        pass
