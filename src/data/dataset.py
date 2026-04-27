"""
Dataset registry and helpers.

`BaseDataset`, `DATASET_FUNCTIONS`, and `randomized_enumerate` live here (Arbor-style hub).
Sibling modules whose names end with `_dataset.py` are imported below so subclasses
register via `__init_subclass__`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from importlib import import_module
from pathlib import Path
from typing import Iterator, Tuple

import numpy as np
from streamlit_app.glossary_types import GlossaryNode

DATASET_FUNCTIONS = {}


class BaseDataset(ABC):
    "this class should be subscriptable"

    def __init__(self):
        self.data = None
        self.index_array = None

    def __init_subclass__(cls, **kwargs):
        display_name = kwargs.pop(
            "display_name",
            cls.__name__.replace("Dataset", "").lower(),
        )
        if kwargs:
            raise TypeError(f"Unexpected keyword arguments for {cls.__name__}: {tuple(kwargs)}")

        cls.display_name = display_name

        if display_name in DATASET_FUNCTIONS:
            raise ValueError(f"Dataset function {display_name} already exists")

        DATASET_FUNCTIONS[display_name] = cls

    def __iter__(self):
        return iter(zip(self.data["X"], self.data["Y"]))

    def __len__(self):
        return len(self.data["X"])

    def __getitem__(
        self, index: int | slice
    ) -> Tuple[np.ndarray, np.ndarray] | Iterator[Tuple[np.ndarray, np.ndarray]]:
        return (self.data["X"][index], self.data["Y"][index])

    def name(self) -> str:
        return self.display_name

    @classmethod
    @abstractmethod
    def get_glossary_node(cls) -> GlossaryNode:
        raise NotImplementedError

    @abstractmethod
    def load_data(self) -> bool:
        raise NotImplementedError("load_data not implemented")

    def get_data(self) -> dict:
        return self.data

    def is_data_loaded(self) -> bool:
        return self.data is not None


def randomized_enumerate(
    dataset: BaseDataset,
) -> Iterator[Tuple[int, Tuple[np.ndarray, np.ndarray]]]:
    index_array = np.arange(len(dataset))
    np.random.shuffle(index_array)
    for index in index_array:
        yield int(index), dataset.__getitem__(index)


for _path in sorted(Path(__file__).resolve().parent.iterdir()):
    if _path.is_dir() or _path.name == "__init__.py":
        continue
    if not _path.name.endswith("_dataset.py"):
        continue
    _stem = _path.stem
    _pkg = __package__
    if _pkg:
        import_module(f".{_stem}", _pkg)
    else:
        import_module(f"data.{_stem}")


def list_registered_dataset_display_names() -> list[str]:
    return sorted(DATASET_FUNCTIONS.keys())


def is_registered_dataset_display_name(display_name: str) -> bool:
    return display_name in DATASET_FUNCTIONS


def build_registered_dataset(
    display_name: str,
    *,
    d: int,
    num_samples: int,
    scale: float | np.float64 = 1.0,
) -> BaseDataset:
    cls = DATASET_FUNCTIONS.get(display_name)
    if cls is None:
        raise ValueError(f"Unknown dataset display_name: {display_name!r}")
    return cls(d=d, num_samples=num_samples, scale=scale)


def build_datasets_glossary_parent() -> GlossaryNode:
    children = tuple(DATASET_FUNCTIONS[name].get_glossary_node() for name in sorted(DATASET_FUNCTIONS.keys()))
    return GlossaryNode(
        title="Datasets",
        aliases=("data", "training data", "samples"),
        english=(
            "A dataset here is an iterable of (input, target) pairs used for training. "
            "Each vector has length **n** (the network’s node count). Expand the entries below "
            "for the kinds used in this project."
        ),
        children=children,
    )
