from abc import ABC, abstractmethod
from typing import List, Iterator, Tuple
import numpy as np

DATASET_FUNCTIONS = {}


class BaseDataset(ABC):
    "this class should be subscriptable"

    def __init__(self):
        self.data = None
        self.index_array = None

    def __init_subclass__(self, **kwargs):
        self.display_name = kwargs.get("display_name", self.__class__.__name__.lower())

        if self.display_name in DATASET_FUNCTIONS:
            raise ValueError(f"Dataset function {self.display_name} already exists")

        DATASET_FUNCTIONS[self.display_name] = self

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
    def __init__(
        self, d: int = 2, num_samples: int = 100, scale: float | np.float64 = 1.0
    ):
        super().__init__()
        self.d = d
        self.num_samples = num_samples
        self.scale = scale
        self.data = None

        self.load_data()

    def load_data(self) -> bool:
        X = (np.random.rand(self.num_samples, self.d) * 2 - 1).astype(float)
        Y = X.copy() * self.scale
        self.data = {
            "X": X,
            "Y": Y,
        }
        return True


class IdentityDataset(LinearScaleDataset, display_name="identity"):
    def __init__(self, d: int = 2, num_samples: int = 100):
        super().__init__(d, num_samples)
        self.scale = 1.0


class DiagonalScaleDataset(LinearScaleDataset, display_name="diagonal_scale"):
    def __init__(self, d: int = 2, num_samples: int = 100):
        super().__init__(d, num_samples)
        self.scale = 1.0
        self.data = None

        self.load_data()

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
