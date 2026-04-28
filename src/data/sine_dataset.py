import numpy as np

from data.dataset import BaseDataset
from hexnets_web.glossary_types import GlossaryNode


class SineDataset(BaseDataset, display_name="sine"):
    def __init__(
        self,
        d: int = 2,
        num_samples: int = 100,
        scale: float | np.float64 = 1.0,
        seed: int | None = None,
        omega: float = 1.0,
    ):
        super().__init__()
        self.d = d
        self.num_samples = num_samples
        self.scale = float(scale)
        self.seed = seed
        self.omega = float(omega)
        self.data = None
        self.load_data()

    @classmethod
    def get_glossary_node(cls) -> GlossaryNode:
        return GlossaryNode(
            title="Sine dataset",
            aliases=("sine", "SineDataset"),
            english=(
                "Inputs **x** are uniform in [-1, 1]^d; targets **y = sin(ω x)** elementwise "
                "(default ω=1). CLI **scale** is accepted but not used."
            ),
            math_latex=r"y = \sin(\omega x)",
            example="Larger ω increases oscillation frequency and can make fitting harder.",
            good_for="Testing approximation capacity; under/overfitting behavior.",
            tags=(
                "deterministic-given-seed",
                "regression-compatible",
                "nonlinear",
                "periodic",
            ),
            children=(),
        )

    def load_data(self) -> bool:
        rng = np.random.default_rng(self.seed)
        X = (rng.random((self.num_samples, self.d)) * 2 - 1).astype(float)
        Y = np.sin(self.omega * X)
        self.data = {"X": X, "Y": Y}
        return True
