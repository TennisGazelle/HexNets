from abc import ABC, abstractmethod
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


class LinearScaleDataset(BaseDataset, display_name="linear_scale"):
    def __init__(self, d: int = 2, num_samples: int = 100, scale: float | np.float64 = 1.0):
        super().__init__()
        self.d = d
        self.num_samples = num_samples
        self.scale = scale
        self.data = None

        self.load_data()

    @classmethod
    def get_glossary_node(cls) -> GlossaryNode:
        return GlossaryNode(
            title="Linear (scaled) dataset",
            aliases=(
                "linear_scale",
                "type=linear_scale",
                "LinearScaleDataset",
                "linear",
                "type=linear",
            ),
            english=(
                "Inputs **x** are drawn uniformly in [-1, 1]^d; targets are **y = scale · x** with a "
                "scalar `scale` (default 2.0 in `hexnet train` for this dataset). CLI and manifests use the id "
                "**linear_scale** (same as the dataset `display_name`). The Streamlit **Train Network** "
                "button uses the identity variant (`get_dataset(..., type='identity')`), not this one."
            ),
            math_latex=r"y = s \cdot x",
            example="With scale=2, if x = [0.5, -1.0] then y = [1.0, -2.0].",
            children=(),
        )

    def load_data(self) -> bool:
        X = (np.random.rand(self.num_samples, self.d) * 2 - 1).astype(float)
        Y = X.copy() * self.scale
        self.data = {
            "X": X,
            "Y": Y,
        }
        return True


class IdentityDataset(LinearScaleDataset, display_name="identity"):
    def __init__(
        self,
        d: int = 2,
        num_samples: int = 100,
        scale: float | np.float64 | None = None,
    ):
        super().__init__(d, num_samples)
        self.scale = 1.0

    @classmethod
    def get_glossary_node(cls) -> GlossaryNode:
        return GlossaryNode(
            title="Identity dataset",
            aliases=("identity", "type=identity"),
            english=(
                "Training pairs where each target **y** equals the corresponding input **x**. "
                "In code this is `IdentityDataset`: it uses the same random **x** in [-1, 1]^d as "
                "`LinearScaleDataset` but with scale 1, so **y = x**."
            ),
            math_latex=r"y = x \quad \text{(elementwise)}",
            example="For d=3, one sample might be x = [0.2, -0.5, 0.1] and y = [0.2, -0.5, 0.1].",
            children=(),
        )


class DiagonalScaleDataset(LinearScaleDataset, display_name="diagonal_scale"):
    def __init__(
        self,
        d: int = 2,
        num_samples: int = 100,
        scale: float | np.float64 = 1.0,
    ):
        super().__init__(d, num_samples, scale)
        self.data = None
        self.load_data()

    @classmethod
    def get_glossary_node(cls) -> GlossaryNode:
        return GlossaryNode(
            title="Diagonal scale dataset",
            aliases=("diagonal_scale", "type=diagonal_scale", "DiagonalScaleDataset"),
            english=(
                "Like the linear scaled dataset, **x** is uniform in [-1, 1]^d, but each output "
                "coordinate is scaled independently: **y_i = s · (i+1) · x_i** for zero-based index **i**. "
                "Useful for per-dimension behavior and conditioning; CLI id **diagonal_scale**."
            ),
            math_latex=r"y_i = s \cdot (i+1) \cdot x_i",
            example=(
                "With d=3 and scale=1, if x = [0.2, 0.5, -0.1] then "
                "y = [0.2, 1.0, -0.3] (dimension i is scaled by (i+1)·s)."
            ),
            children=(),
        )

    def load_data(self) -> bool:
        X = (np.random.rand(self.num_samples, self.d) * 2 - 1).astype(float)
        Y = X.copy()
        for i in range(self.d):
            Y[:, i] *= (i + 1) * self.scale
        self.data = {
            "X": X,
            "Y": Y,
        }
        return True


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
