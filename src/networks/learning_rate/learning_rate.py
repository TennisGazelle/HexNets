from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from networks.learning_rate import LEARNING_RATES
from streamlit_app.glossary_types import GlossaryNode


class BaseLearningRate(ABC):
    def __init_subclass__(self, **kwargs):
        self.display_name = kwargs.get("display_name", self.__class__.__name__.lower())

        if self.display_name in LEARNING_RATES:
            raise ValueError(f"Learning rate {self.display_name} already exists")

        LEARNING_RATES[self.display_name] = self

    def __str__(self):
        return self.display_name

    def __repr__(self):
        return self.display_name

    @classmethod
    @abstractmethod
    def get_glossary_node(cls) -> GlossaryNode:
        raise NotImplementedError

    @abstractmethod
    def rate_at_iteration(self, iteration: int) -> float:
        raise NotImplementedError("rate_at_iteration not implemented")


def get_learning_rate(display_name: str, *args, **kwargs) -> BaseLearningRate:
    if display_name not in LEARNING_RATES:
        raise ValueError(f"Learning rate {display_name} not found")
    return LEARNING_RATES[display_name](*args, **kwargs)


def get_available_learning_rates() -> List[str]:
    return list(LEARNING_RATES.keys())


def build_learning_rates_glossary_parent() -> GlossaryNode:
    children = tuple(LEARNING_RATES[name].get_glossary_node() for name in sorted(LEARNING_RATES.keys()))
    return GlossaryNode(
        title="Learning rates",
        aliases=("lr", "step size", "LEARNING_RATES", "learning rate schedule"),
        english=(
            "The optimizer multiplies gradients by a scalar **learning rate** each update. Here a **schedule** "
            "implements **rate_at_iteration(iteration)** so the effective step size can stay constant or decay over "
            "training. The Streamlit **Learning rate** control picks one of the registered schedules below."
        ),
        children=children,
    )
