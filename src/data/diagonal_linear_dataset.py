import numpy as np

from data.dataset import BaseDataset, DatasetNoiseMode
from hexnets_web.glossary_types import GlossaryNode


class DiagonalLinearDataset(BaseDataset, display_name="diagonal_linear"):
    def __init__(
        self,
        d: int = 2,
        num_samples: int = 100,
        scale: float | np.float64 = 1.0,
        seed: int | None = None,
        min_scale: float = 0.5,
        max_scale: float = 2.0,
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
        self.min_scale = float(min_scale)
        self.max_scale = float(max_scale)
        seed_i = 0 if self.seed is None else int(self.seed) & 0xFFFFFFFF
        parent = np.random.SeedSequence([seed_i, 0xD1A60, self.d, self.num_samples])
        s_a, s_x = parent.spawn(2)
        rng_a = np.random.default_rng(s_a)
        self._a = rng_a.uniform(self.min_scale, self.max_scale, size=(self.d,))
        self._rng_inputs = np.random.default_rng(s_x)
        self.data = None
        self.load_data()

    @classmethod
    def get_glossary_node(cls) -> GlossaryNode:
        return GlossaryNode(
            title="Diagonal linear dataset",
            aliases=("diagonal_linear", "DiagonalLinearDataset"),
            english=(
                "Inputs **x** are uniform in [-1, 1]^d. A random diagonal **a** is drawn once per "
                "dataset (per dimension in [min_scale, max_scale]); targets are **y_i = a_i x_i**. "
                "CLI **scale** is not used for this mapping (see **full_linear** to tune operator gain)."
            ),
            math_latex=r"y_i = a_i x_i,\quad a_i \sim \mathrm{Unif}(\texttt{min\_scale},\texttt{max\_scale})",
            example="With d=2, a might be [0.8, 1.4]; then y = [0.8·x₀, 1.4·x₁] elementwise.",
            good_for="Per-dimension learning behavior; Adam vs SGD comparisons.",
            tags=(
                "deterministic-given-seed",
                "regression-compatible",
                "linear",
                "random-diagonal",
            ),
            children=(),
        )

    def _sample_inputs_rng_impl(self, **kwargs) -> np.ndarray:
        return (self._rng_inputs.random((self.num_samples, self.d)) * 2 - 1).astype(float)

    def targets_from_inputs(self, X: np.ndarray) -> np.ndarray:
        x = self._as_validated_batch_inputs(X)
        return x * self._a
