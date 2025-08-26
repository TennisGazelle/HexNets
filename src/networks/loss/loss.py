from abc import ABC, abstractmethod

from src.networks.loss import LOSS_FUNCTIONS


class BaseLoss(ABC):

    def __init_subclass__(self, **kwargs):
        self.display_name = kwargs.get("display_name", self.__class__.__name__.lower())

        if self.display_name in LOSS_FUNCTIONS:
            raise ValueError(f"Loss function {self.display_name} already exists")

        LOSS_FUNCTIONS[self.display_name] = self

    @abstractmethod
    def calc_loss(self, y_true, y_pred):
        raise NotImplementedError("calc_loss not implemented")

    @abstractmethod
    def calc_delta(self, y_true, y_pred):
        raise NotImplementedError("calc_delta not implemented")

    def __str__(self):
        return self.display_name

    def __repr__(self):
        return self.display_name


def get_loss_function(display_name: str, *args, **kwargs) -> BaseLoss:
    if display_name not in LOSS_FUNCTIONS:
        raise ValueError(f"Loss function {display_name} not found")
    return LOSS_FUNCTIONS[display_name](*args, **kwargs)


def get_available_loss_functions() -> list[str]:
    return list(LOSS_FUNCTIONS.keys())
