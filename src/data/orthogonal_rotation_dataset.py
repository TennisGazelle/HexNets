import numpy as np

from data.dataset import BaseDataset
from hexnets_web.glossary_types import GlossaryNode


class OrthogonalRotationDataset(BaseDataset, display_name="orthogonal_rotation"):
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
            title="Orthogonal rotation dataset",
            aliases=("orthogonal_rotation", "OrthogonalRotationDataset", "qr rotation"),
            english=(
                "Inputs **x** are standard normal; **Q** is a random orthogonal matrix from the QR factorization "
                "of a Gaussian matrix; **y = x Q^T**. Norms are preserved (||y|| = ||x|| row-wise). "
                "CLI **scale** is accepted for registry uniformity but **not applied** to this map."
            ),
            math_latex=r"y = x Q^\top,\quad Q^\top Q = I",
            example="Each row of Y is a rigid rotation/reflection of the corresponding row of X.",
            good_for="Testing invariances; cosine similarity losses; stable transforms.",
            tags=(
                "deterministic-given-seed",
                "regression-compatible",
                "orthogonal",
                "norm-preserving",
            ),
            children=(),
        )

    def load_data(self) -> bool:
        rng = np.random.default_rng(self.seed)
        X = rng.standard_normal((self.num_samples, self.d)).astype(float)
        M = rng.standard_normal((self.d, self.d))
        Q, _ = np.linalg.qr(M)
        Y = X @ Q.T
        self.data = {"X": X, "Y": Y}
        return True
