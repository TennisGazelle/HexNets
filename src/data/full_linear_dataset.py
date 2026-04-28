import numpy as np

from data.dataset import BaseDataset
from hexnets_web.glossary_types import GlossaryNode


class FullLinearDataset(BaseDataset, display_name="full_linear"):
    def __init__(
        self,
        d: int = 2,
        num_samples: int = 100,
        scale: float | np.float64 = 1.0,
        seed: int | None = None,
    ):
        super().__init__()
        self.d = d
        self.num_samples = num_samples
        self.A_scale = float(scale)
        self.seed = seed
        self.data = None
        self.load_data()

    @classmethod
    def get_glossary_node(cls) -> GlossaryNode:
        return GlossaryNode(
            title="Full linear dataset",
            aliases=("full_linear", "FullLinearDataset"),
            english=(
                "Inputs **x** are standard normal in R^d; **A** is a random d×d matrix scaled by "
                "CLI **scale** (story **A_scale**); targets **y = x A^T** (row-vector convention matching "
                "the story’s `X @ A.T`)."
            ),
            math_latex=r"y = x A^\top",
            example="If scale=0.5, entries of A are N(0, 0.5²) so the operator is milder.",
            good_for="Learning cross-feature coupling; stress-testing depth/width.",
            tags=(
                "deterministic-given-seed",
                "regression-compatible",
                "linear",
                "dense-operator",
            ),
            children=(),
        )

    def load_data(self) -> bool:
        rng = np.random.default_rng(self.seed)
        X = rng.standard_normal((self.num_samples, self.d)).astype(float)
        A = rng.standard_normal((self.d, self.d)).astype(float) * self.A_scale
        Y = X @ A.T
        self.data = {"X": X, "Y": Y}
        return True
