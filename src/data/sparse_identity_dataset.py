import numpy as np

from data.dataset import BaseDataset
from streamlit_app.glossary_types import GlossaryNode


class SparseIdentityDataset(BaseDataset, display_name="sparse_identity"):
    def __init__(
        self,
        d: int = 2,
        num_samples: int = 100,
        scale: float | np.float64 = 1.0,
        seed: int | None = None,
        k_nonzero: int | None = None,
    ):
        super().__init__()
        self.d = d
        self.num_samples = num_samples
        self.scale = float(scale)
        self.seed = seed
        default_k = min(3, max(self.d, 1))
        self.k_nonzero = default_k if k_nonzero is None else int(k_nonzero)
        self.k_nonzero = max(1, min(self.k_nonzero, max(self.d, 1)))
        self.data = None
        self.load_data()

    @classmethod
    def get_glossary_node(cls) -> GlossaryNode:
        return GlossaryNode(
            title="Sparse identity dataset",
            aliases=("sparse_identity", "SparseIdentityDataset"),
            english=(
                "Each sample **x** has at most **k_nonzero** nonzero coordinates (default min(3, d)), "
                "drawn from a standard normal at random indices; **y = x**. Most entries are zero. "
                "CLI **scale** is accepted but not used."
            ),
            math_latex=r"y = x,\quad x\ \text{sparse}",
            example="Useful when many input dims are inactive but a few carry signal.",
            good_for="Sparse regimes; Adam vs SGD; handling many zeros.",
            tags=(
                "deterministic-given-seed",
                "regression-compatible",
                "linear",
                "sparse",
            ),
            children=(),
        )

    def load_data(self) -> bool:
        rng = np.random.default_rng(self.seed)
        k = self.k_nonzero
        X = np.zeros((self.num_samples, self.d), dtype=float)
        idx = np.stack([rng.choice(self.d, size=k, replace=False) for _ in range(self.num_samples)])
        vals = rng.standard_normal((self.num_samples, k))
        X[np.arange(self.num_samples)[:, None], idx] = vals
        Y = X.copy()
        self.data = {"X": X, "Y": Y}
        return True
