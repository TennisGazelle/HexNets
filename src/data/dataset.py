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
from typing import Iterator, Literal, Tuple

import numpy as np
from hexnets_web.glossary_types import GlossaryNode

DATASET_FUNCTIONS = {}

DatasetNoiseMode = Literal["inputs", "targets", "both"]
_NOISE_SEED_ENTROPY_TAG = 0x4E015E  # tag for isolated dataset-noise RNG stream

# --- static methods ---


def _dataset_noise_entropy(noise_seed: int) -> list[int]:
    """Entropy list for an RNG stream independent of global ``np.random``."""
    return [int(noise_seed) & 0xFFFFFFFF, _NOISE_SEED_ENTROPY_TAG]


class BaseDataset(ABC):
    "this class should be subscriptable"

    def __init__(
        self,
        *,
        noise_mode: DatasetNoiseMode | None = None,
        noise_mu: float = 0.0,
        noise_sigma: float = 0.1,
        noise_seed: int = 0,
    ):
        self.data = None
        self.index_array = None
        self.noise_mode: DatasetNoiseMode | None = noise_mode
        self.noise_mu = float(noise_mu)
        self.noise_sigma = float(noise_sigma)
        self.noise_seed = int(noise_seed)

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
    def _load_data_impl(self) -> None:
        raise NotImplementedError("_load_data_impl not implemented")

    def load_data(self) -> bool:
        self._load_data_impl()
        self._apply_configured_gaussian_noise()
        return True

    def _apply_configured_gaussian_noise(self) -> None:
        if self.noise_mode is None or self.data is None:
            return
        if self.noise_sigma < 0:
            raise ValueError("noise_sigma must be non-negative")
        rng = np.random.default_rng(np.random.SeedSequence(_dataset_noise_entropy(self.noise_seed)))
        mode = self.noise_mode
        mu, sigma = self.noise_mu, self.noise_sigma

        if mode in ("inputs", "both"):
            X = self.data["X"]
            noise_x = rng.normal(loc=mu, scale=sigma, size=X.shape)
            self.data["X"] = (X.astype(np.float64, copy=False) + noise_x).astype(X.dtype, copy=False)

        if mode in ("targets", "both"):
            Y = self.data["Y"]
            noise_y = rng.normal(loc=mu, scale=sigma, size=Y.shape)
            self.data["Y"] = (Y.astype(np.float64, copy=False) + noise_y).astype(Y.dtype, copy=False)

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
    noise_mode: DatasetNoiseMode | None = None,
    noise_mu: float = 0.0,
    noise_sigma: float = 0.1,
    noise_seed: int = 0,
) -> BaseDataset:
    cls = DATASET_FUNCTIONS.get(display_name)
    if cls is None:
        raise ValueError(f"Unknown dataset display_name: {display_name!r}")
    return cls(
        d=d,
        num_samples=num_samples,
        scale=scale,
        noise_mode=noise_mode,
        noise_mu=noise_mu,
        noise_sigma=noise_sigma,
        noise_seed=noise_seed,
    )


def build_datasets_glossary_parent() -> GlossaryNode:
    children = tuple(DATASET_FUNCTIONS[name].get_glossary_node() for name in sorted(DATASET_FUNCTIONS.keys()))
    return GlossaryNode(
        title="Datasets",
        aliases=("data", "training data", "samples"),
        english=(
            "A dataset here is an iterable of (input, target) pairs used for training. "
            "Each vector has length **n** (the network's node count). Expand the entries below "
            "for the kinds used in this project. Many entries include **tags** and a **Good for** line "
            "when the glossary node defines them."
        ),
        children=children,
    )
