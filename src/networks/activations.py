from abc import ABC, abstractmethod
import numpy as np

class BaseActivation(ABC):
    def __init__(self, name):
        self.name = name

    @abstractmethod
    def activate(self, x):
        pass

    @abstractmethod
    def deactivate(self, x):
        pass


class ReLU(BaseActivation):
    def __init__(self):
        super().__init__("relu")

    def activate(self, x):
        return np.maximum(0, x)

    def deactivate(self, x):
        return (x > 0).astype(float)

class LeakyReLU(BaseActivation):
    def __init__(self, alpha=0.01):
        super().__init__("leaky_relu")
        self.alpha = alpha

    def activate(self, x):
        return np.where(x > 0, x, self.alpha * x)

    def deactivate(self, x):
        return np.where(x > 0, 1, self.alpha)

class Sigmoid(BaseActivation):
    def __init__(self):
        super().__init__("sigmoid")

    def activate(self, x):
        return 1 / (1 + np.exp(-x))

    def deactivate(self, x):
        return self.activate(x) * (1 - self.activate(x))