from abc import ABC, abstractmethod

# === Base class ===
class BaseNeuralNetwork(ABC):
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
