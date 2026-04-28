import numpy as np

from data.dataset import BaseDataset
from hexnets_web.glossary_types import GlossaryNode


class AffineDataset(BaseDataset, display_name="affine"):
    def __init__(
        self,
        d: int = 2,
        num_samples: int = 100,
        scale: float | np.float64 = 1.0,
        seed: int | None = None,
        b_scale: float = 1.0,
    ):
        super().__init__()
        self.d = d
        self.num_samples = num_samples
        self.A_scale = float(scale)
        self.seed = seed
        self.b_scale = float(b_scale)
        self.data = None
        self.load_data()

    @classmethod
    def get_glossary_node(cls) -> GlossaryNode:
        return GlossaryNode(
            title="Affine dataset",
            aliases=("affine", "AffineDataset"),
            english=(
                "Inputs **x** are standard normal; random **A** (scaled by CLI **scale** as **A_scale**) "
                "and random bias **b** with coordinates uniform in [-b_scale, b_scale]; **y = x A^T + b**."
            ),
            math_latex=r"y = x A^\top + b",
            example="Nonzero **b** checks that the network can learn offsets, not only homogeneous maps.",
            good_for="Ensuring your model has bias terms and learns offsets.",
            tags=(
                "deterministic-given-seed",
                "regression-compatible",
                "affine-operator",
                "translation",
            ),
            children=(),
        )

    def load_data(self) -> bool:
        rng = np.random.default_rng(self.seed)
        X = rng.standard_normal((self.num_samples, self.d)).astype(float)
        A = rng.standard_normal((self.d, self.d)).astype(float) * self.A_scale
        b = rng.uniform(-self.b_scale, self.b_scale, size=(self.d,)).astype(float)
        Y = X @ A.T + b
        self.data = {"X": X, "Y": Y}
        return True
