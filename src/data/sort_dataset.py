import numpy as np

from data.dataset import BaseDataset, DatasetNoiseMode
from hexnets_web.glossary_types import GlossaryNode


class SortDataset(BaseDataset, display_name="sort"):
    def __init__(
        self,
        d: int = 2,
        num_samples: int = 100,
        scale: float | np.float64 = 1.0,
        seed: int | None = None,
        *,
        noise_mode: DatasetNoiseMode | None = None,
        noise_mu: float = 0.0,
        noise_sigma: float = 0.1,
        noise_seed: int = 0,
    ):
        super().__init__(
            noise_mode=noise_mode,
            noise_mu=noise_mu,
            noise_sigma=noise_sigma,
            noise_seed=noise_seed,
        )
        self.d = d
        self.num_samples = num_samples
        self.scale = float(scale)
        self.seed = seed
        self.data = None
        self.load_data()

    @classmethod
    def get_glossary_node(cls) -> GlossaryNode:
        return GlossaryNode(
            title="Sort dataset",
            aliases=("sort", "SortDataset"),
            english=(
                "Inputs **x** are uniform in [-1, 1]^d; targets **y** are **x** sorted ascending along "
                "each row. CLI **scale** is accepted but not used."
            ),
            math_latex=r"y = \mathrm{sort}(x)\ \text{(ascending per row)}",
            example="Hard, highly structured map; useful for stress-testing capacity.",
            good_for="Hard-mode; shows limits of standard feedforward nets quickly.",
            tags=(
                "deterministic-given-seed",
                "regression-compatible",
                "non-smooth",
                "structured",
                "stress-test",
            ),
            children=(),
        )

    def _load_data_impl(self) -> None:
        rng = np.random.default_rng(self.seed)
        X = (rng.random((self.num_samples, self.d)) * 2 - 1).astype(float)
        Y = np.sort(X, axis=1)
        self.data = {"X": X, "Y": Y}
