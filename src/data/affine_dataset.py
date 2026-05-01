import numpy as np

from data.dataset import BaseDataset, DatasetNoiseMode
from hexnets_web.glossary_types import GlossaryNode


class AffineDataset(BaseDataset, display_name="affine"):
    def __init__(
        self,
        d: int = 2,
        num_samples: int = 100,
        scale: float | np.float64 = 1.0,
        seed: int | None = None,
        b_scale: float = 1.0,
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
        self.A_scale = float(scale)
        self.seed = seed
        self.b_scale = float(b_scale)
        seed_i = 0 if self.seed is None else int(self.seed) & 0xFFFFFFFF
        parent = np.random.SeedSequence([seed_i, 0xAF001E, self.d, self.num_samples])
        s_mat, s_x = parent.spawn(2)
        rng_mat = np.random.default_rng(s_mat)
        self._A = rng_mat.standard_normal((self.d, self.d)).astype(float) * self.A_scale
        self._b = rng_mat.uniform(-self.b_scale, self.b_scale, size=(self.d,)).astype(float)
        self._rng_inputs = np.random.default_rng(s_x)
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

    def _sample_inputs_rng_impl(self, **kwargs) -> np.ndarray:
        return self._rng_inputs.standard_normal((self.num_samples, self.d)).astype(float)

    def targets_from_inputs(self, X: np.ndarray) -> np.ndarray:
        x = self._as_validated_batch_inputs(X)
        return x @ self._A.T + self._b
