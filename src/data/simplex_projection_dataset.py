import numpy as np

from data.dataset import BaseDataset
from streamlit_app.glossary_types import GlossaryNode


class SimplexProjectionDataset(BaseDataset, display_name="simplex_projection"):
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

    @staticmethod
    def _proj_simplex(v: np.ndarray) -> np.ndarray:
        """Euclidean projection of vector v onto the probability simplex."""
        n = v.size
        u = np.sort(v)[::-1]
        cssv = np.cumsum(u)
        idx = np.arange(1, n + 1, dtype=float)
        cond = u * idx > (cssv - 1)
        rho = int(np.nonzero(cond)[0][-1])
        theta = (cssv[rho] - 1.0) / (rho + 1)
        return np.maximum(v - theta, 0.0)

    @classmethod
    def get_glossary_node(cls) -> GlossaryNode:
        return GlossaryNode(
            title="Simplex projection dataset",
            aliases=("simplex_projection", "SimplexProjectionDataset"),
            english=(
                "Inputs **x** are standard normal; each row is projected onto the **probability simplex** "
                "(nonnegative entries summing to 1). CLI **scale** is accepted but not used."
            ),
            math_latex=r"y = \Pi_\Delta(x)",
            example="Each row of Y is a probability vector suitable for KL-style objectives.",
            good_for="“Distribution” outputs; KL-divergence losses; constrained learning.",
            tags=(
                "deterministic-given-seed",
                "regression-compatible",
                "nonlinear",
                "projection",
                "probability-simplex",
            ),
            children=(),
        )

    def load_data(self) -> bool:
        rng = np.random.default_rng(self.seed)
        X = rng.standard_normal((self.num_samples, self.d)).astype(float)
        Y = np.stack([self._proj_simplex(X[i]) for i in range(self.num_samples)], axis=0)
        self.data = {"X": X, "Y": Y}
        return True
