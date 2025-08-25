from abc import ABC, abstractmethod


class BaseActivation(ABC):
    def __init__(self, name):
        self.name = name

    @abstractmethod
    def activate(self, x):
        pass

    @abstractmethod
    def deactivate(self, x):
        pass

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name
