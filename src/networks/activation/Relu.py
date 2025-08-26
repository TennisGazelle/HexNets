import numpy as np

from src.networks.activation.activations import BaseActivation


class ReLU(BaseActivation, display_name="relu"):

    def activate(self, x):
        return np.maximum(0, x)

    def deactivate(self, x):
        return (x > 0).astype(float)
