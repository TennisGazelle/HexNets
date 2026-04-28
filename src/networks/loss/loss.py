from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from networks.loss import LOSS_FUNCTIONS
from hexnets_web.glossary_types import GlossaryNode


class BaseLoss(ABC):

    def __init_subclass__(self, **kwargs):
        self.display_name = kwargs.get("display_name", self.__class__.__name__.lower())

        if self.display_name in LOSS_FUNCTIONS:
            raise ValueError(f"Loss function {self.display_name} already exists")

        LOSS_FUNCTIONS[self.display_name] = self

    def __str__(self):
        return self.display_name

    def __repr__(self):
        return self.display_name

    @classmethod
    @abstractmethod
    def get_glossary_node(cls) -> GlossaryNode:
        raise NotImplementedError

    @abstractmethod
    def calc_loss(self, y_true, y_pred):
        raise NotImplementedError("calc_loss not implemented")

    @abstractmethod
    def calc_delta(self, y_true, y_pred):
        raise NotImplementedError("calc_delta not implemented")


def get_loss_function(display_name: str, *args, **kwargs) -> BaseLoss:
    if display_name not in LOSS_FUNCTIONS:
        raise ValueError(f"Loss function {display_name} not found")
    return LOSS_FUNCTIONS[display_name](*args, **kwargs)


def get_available_loss_functions() -> List[str]:
    return list(LOSS_FUNCTIONS.keys())


def build_losses_glossary_parent() -> GlossaryNode:
    children = tuple(LOSS_FUNCTIONS[name].get_glossary_node() for name in sorted(LOSS_FUNCTIONS.keys()))
    return GlossaryNode(
        title="Loss functions",
        aliases=("loss", "objective", "training loss", "LOSS_FUNCTIONS"),
        english=(
            "A **loss class** defines how the network scores predictions **y_pred** against targets **y_true** "
            "and what signal is backpropagated via **calc_delta**. The trainer averages the per-sample loss over "
            "the batch/epoch for logging (see **Loss (epoch)** under **Training metrics**). Expand the entries below "
            "for each implementation available in this project."
        ),
        children=children,
    )
