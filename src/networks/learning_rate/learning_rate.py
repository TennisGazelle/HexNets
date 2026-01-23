from abc import ABC, abstractmethod
from networks.learning_rate import LEARNING_RATES
import numpy as np
from typing import List


class BaseLearningRate(ABC):
    def __init_subclass__(self, **kwargs):
        self.display_name = kwargs.get("display_name", self.__class__.__name__.lower())

        if self.display_name in LEARNING_RATES:
            raise ValueError(f"Learning rate {self.display_name} already exists")

        LEARNING_RATES[self.display_name] = self

    @abstractmethod
    def rate_at_iteration(self, iteration: int) -> float:
        raise NotImplementedError("rate_at_iteration not implemented")

    def __str__(self):
        return self.display_name

    def __repr__(self):
        return self.display_name


def get_learning_rate(display_name: str, *args, **kwargs) -> BaseLearningRate:
    if display_name not in LEARNING_RATES:
        raise ValueError(f"Learning rate {display_name} not found")
    return LEARNING_RATES[display_name](*args, **kwargs)


def get_available_learning_rates() -> List[str]:
    return list(LEARNING_RATES.keys())
