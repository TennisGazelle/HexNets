import numpy as np

from data.dataset import BaseDataset
from streamlit_app.glossary_types import GlossaryNode


class LowRankLinearDataset(BaseDataset, display_name="low_rank_linear"):
    def __init__(
        self,
        d: int = 2,
        num_samples: int = 100,
        scale: float | np.float64 = 1.0,
        seed: int | None = None,
        rank: int = 2,
    ):
        super().__init__()
        self.d = d
        self.num_samples = num_samples
        self.gain = float(scale)
        self.seed = seed
        self.rank = max(1, min(int(rank), max(self.d, 1)))
        self.data = None
        self.load_data()

    @classmethod
    def get_glossary_node(cls) -> GlossaryNode:
        return GlossaryNode(
            title="Low-rank linear dataset",
            aliases=("low_rank_linear", "LowRankLinearDataset"),
            english=(
                "Inputs **x** are standard normal; **A = U V^T** with **U,V** ∈ R^{d×rank} Gaussian; "
                "targets **y = x A^T** scaled by CLI **scale** as an overall gain on **A**. "
                "Python parameter **rank** (default 2) is unrelated to hex rotation **r**."
            ),
            math_latex=r"y = x (U V^\top)^\top,\quad \mathrm{rank}(UV^\top)\le \texttt{rank}",
            example="Structured coupling with lower rank than a full dense d×d linear map.",
            good_for="Structured coupling; generalization vs memorization.",
            tags=(
                "deterministic-given-seed",
                "regression-compatible",
                "linear",
                "low-rank",
            ),
            children=(),
        )

    def load_data(self) -> bool:
        rng = np.random.default_rng(self.seed)
        X = rng.standard_normal((self.num_samples, self.d)).astype(float)
        r = self.rank
        U = rng.standard_normal((self.d, r)).astype(float)
        V = rng.standard_normal((self.d, r)).astype(float)
        A = (U @ V.T) * self.gain
        Y = X @ A.T
        self.data = {"X": X, "Y": Y}
        return True
