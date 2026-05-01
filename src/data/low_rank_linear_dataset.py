import numpy as np

from data.dataset import BaseDataset, DatasetNoiseMode
from hexnets_web.glossary_types import GlossaryNode


class LowRankLinearDataset(BaseDataset, display_name="low_rank_linear"):
    def __init__(
        self,
        d: int = 2,
        num_samples: int = 100,
        scale: float | np.float64 = 1.0,
        seed: int | None = None,
        rank: int = 2,
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
        self.gain = float(scale)
        self.seed = seed
        self.rank = max(1, min(int(rank), max(self.d, 1)))
        seed_i = 0 if self.seed is None else int(self.seed) & 0xFFFFFFFF
        parent = np.random.SeedSequence([seed_i, 0x10F2, self.d, self.num_samples, self.rank])
        s_uv, s_x = parent.spawn(2)
        rng_uv = np.random.default_rng(s_uv)
        r = self.rank
        u = rng_uv.standard_normal((self.d, r)).astype(float)
        v = rng_uv.standard_normal((self.d, r)).astype(float)
        self._A = (u @ v.T) * self.gain
        self._rng_inputs = np.random.default_rng(s_x)
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

    def _sample_inputs_rng_impl(self, **kwargs) -> np.ndarray:
        return self._rng_inputs.standard_normal((self.num_samples, self.d)).astype(float)

    def targets_from_inputs(self, X: np.ndarray) -> np.ndarray:
        x = self._as_validated_batch_inputs(X)
        return x @ self._A.T
