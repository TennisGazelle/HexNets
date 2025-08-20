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