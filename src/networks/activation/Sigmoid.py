import numpy as np

from src.networks.activation.activations import BaseActivation


class Sigmoid(BaseActivation):
    def __init__(self):
        super().__init__("sigmoid")

    def activate(self, x):
        return 1 / (1 + np.exp(-x))

    def deactivate(self, x):
        return self.activate(x) * (1 - self.activate(x))
