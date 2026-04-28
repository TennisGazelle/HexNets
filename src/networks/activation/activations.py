from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from networks.activation import ACTIVATION_FUNCTIONS
from hexnets_web.glossary_types import GlossaryNode


class BaseActivation(ABC):

    def __init_subclass__(self, **kwargs):
        self.display_name = kwargs.get("display_name", self.__class__.__name__.lower())

        if self.display_name in ACTIVATION_FUNCTIONS:
            raise ValueError(f"Activation function {self.display_name} already exists")

        ACTIVATION_FUNCTIONS[self.display_name] = self

    def __str__(self):
        return self.display_name

    def __repr__(self):
        return self.display_name

    @classmethod
    @abstractmethod
    def get_glossary_node(cls) -> GlossaryNode:
        raise NotImplementedError

    @abstractmethod
    def activate(self, x):
        raise NotImplementedError("activate not implemented")

    @abstractmethod
    def deactivate(self, x):
        raise NotImplementedError("deactivate not implemented")


def get_activation_function(display_name: str, *args, **kwargs) -> BaseActivation:
    if display_name not in ACTIVATION_FUNCTIONS:
        raise ValueError(f"Activation function {display_name} not found")
    return ACTIVATION_FUNCTIONS[display_name](*args, **kwargs)


def get_available_activation_functions() -> List[str]:
    return list(ACTIVATION_FUNCTIONS.keys())


def build_activations_glossary_parent() -> GlossaryNode:
    children = tuple(ACTIVATION_FUNCTIONS[name].get_glossary_node() for name in sorted(ACTIVATION_FUNCTIONS.keys()))
    return GlossaryNode(
        title="Activations",
        aliases=("activation", "nonlinearity", "ACTIVATION_FUNCTIONS", "hidden activation"),
        english=(
            "Hidden layers apply **activate(x)** forward and use **deactivate(x)** (derivative w.r.t. pre-activation) "
            "in the backward pass. The **Activation** dropdown in Network Explorer selects one of the registered "
            "functions below."
        ),
        children=children,
    )
