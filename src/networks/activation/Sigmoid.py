import numpy as np

from src.networks.activation.activations import BaseActivation


class Sigmoid(BaseActivation, display_name="sigmoid"):

    def activate(self, x):
        return 1 / (1 + np.exp(-x))

    def deactivate(self, x):
        return self.activate(x) * (1 - self.activate(x))
