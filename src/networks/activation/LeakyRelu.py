import numpy as np

from src.networks.activation.activations import BaseActivation


class LeakyReLU(BaseActivation):
    def __init__(self, alpha=0.01):
        super().__init__("leaky_relu")
        self.alpha = alpha

    def activate(self, x):
        return np.where(x > 0, x, self.alpha * x)

    def deactivate(self, x):
        return np.where(x > 0, 1, self.alpha)
