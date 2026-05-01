import numpy as np

from data.dataset import BaseDataset, DatasetNoiseMode
from hexnets_web.glossary_types import GlossaryNode


class FullLinearDataset(BaseDataset, display_name="full_linear"):
    def __init__(
        self,
        d: int = 2,
        num_samples: int = 100,
        scale: float | np.float64 = 1.0,
        seed: int | None = None,
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
        seed_i = 0 if self.seed is None else int(self.seed) & 0xFFFFFFFF
        parent = np.random.SeedSequence([seed_i, 0xF0111E, self.d, self.num_samples])
        s_mat, s_x = parent.spawn(2)
        rng_mat = np.random.default_rng(s_mat)
        self._A = rng_mat.standard_normal((self.d, self.d)).astype(float) * self.A_scale
        self._rng_inputs = np.random.default_rng(s_x)
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
                "the story.s `X @ A.T`)."
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

    def _sample_inputs_rng_impl(self, **kwargs) -> np.ndarray:
        return self._rng_inputs.standard_normal((self.num_samples, self.d)).astype(float)

    def targets_from_inputs(self, X: np.ndarray) -> np.ndarray:
        x = self._as_validated_batch_inputs(X)
        return x @ self._A.T
