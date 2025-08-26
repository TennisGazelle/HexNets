from abc import ABC, abstractmethod

from src.networks.activation import ACTIVATION_FUNCTIONS


class BaseActivation(ABC):

    def __init_subclass__(self, **kwargs):
        self.display_name = kwargs.get("display_name", self.__class__.__name__.lower())

        if self.display_name in ACTIVATION_FUNCTIONS:
            raise ValueError(f"Activation function {self.display_name} already exists")

        ACTIVATION_FUNCTIONS[self.display_name] = self

    @abstractmethod
    def activate(self, x):
        raise NotImplementedError("activate not implemented")

    @abstractmethod
    def deactivate(self, x):
        raise NotImplementedError("deactivate not implemented")

    def __str__(self):
        return self.display_name

    def __repr__(self):
        return self.display_name


def get_activation_function(display_name: str, *args, **kwargs) -> BaseActivation:
    if display_name not in ACTIVATION_FUNCTIONS:
        raise ValueError(f"Activation function {display_name} not found")
    return ACTIVATION_FUNCTIONS[display_name](*args, **kwargs)


def get_available_activation_functions() -> list[str]:
    return list(ACTIVATION_FUNCTIONS.keys())
