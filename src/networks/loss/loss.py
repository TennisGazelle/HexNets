from abc import ABC, abstractmethod


class BaseLoss(ABC):
    def __init__(self, name):
        self.name = name

    @abstractmethod
    def calc_loss(self, y_true, y_pred):
        pass

    @abstractmethod
    def calc_delta(self, y_true, y_pred):
        pass

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name
