import numpy as np

from data.dataset import BaseDataset
from streamlit_app.glossary_types import GlossaryNode


class NonNegativeProjectionDataset(BaseDataset, display_name="non_negative_projection"):
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
        self.scale = float(scale)
        self.seed = seed
        self.data = None
        self.load_data()

    @classmethod
    def get_glossary_node(cls) -> GlossaryNode:
        return GlossaryNode(
            title="Non-negative projection dataset",
            aliases=("non_negative_projection", "NonNegativeProjectionDataset", "relu target"),
            english=(
                "Inputs **x** are standard normal; targets **y = max(x, 0)** elementwise (projection onto "
                "the nonnegative orthant). CLI **scale** is accepted but not used."
            ),
            math_latex=r"y = \max(x, 0)",
            example="Negative coordinates of X become 0 in Y; positive coordinates are unchanged.",
            good_for="Piecewise-linear learning; ReLU output experiments.",
            tags=(
                "deterministic-given-seed",
                "regression-compatible",
                "nonlinear",
                "projection",
                "piecewise-linear",
            ),
            children=(),
        )

    def load_data(self) -> bool:
        rng = np.random.default_rng(self.seed)
        X = rng.standard_normal((self.num_samples, self.d)).astype(float)
        Y = np.maximum(X, 0.0)
        self.data = {"X": X, "Y": Y}
        return True
