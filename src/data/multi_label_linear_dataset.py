import numpy as np

from data.dataset import BaseDataset, DatasetNoiseMode
from hexnets_web.glossary_types import GlossaryNode


class MultiLabelFromLinearDataset(BaseDataset, display_name="multi_label_linear"):
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
        parent = np.random.SeedSequence([seed_i, 0xE11B, self.d, self.num_samples])
        s_ab, s_x = parent.spawn(2)
        rng_ab = np.random.default_rng(s_ab)
        self._A = rng_ab.standard_normal((self.d, self.d)).astype(float) * self.A_scale
        self._b = rng_ab.standard_normal((self.d,)).astype(float) * self.b_scale
        self._rng_inputs = np.random.default_rng(s_x)
        self.data = None
        self.load_data()

    @classmethod
    def get_glossary_node(cls) -> GlossaryNode:
        return GlossaryNode(
            title="Multi-label from linear dataset",
            aliases=("multi_label_linear", "MultiLabelFromLinearDataset"),
            english=(
                "Inputs **x** are standard normal; random **A** (scaled by CLI **scale**) and **b** "
                "(Gaussian times **b_scale**); logits **z = x A^T + b**; targets **y = 1[z > 0]** as floats. "
                "Correlated multi-label structure across dimensions."
            ),
            math_latex=r"y = \mathbb{1}[x A^\top + b > 0]",
            example="Unlike independent per-dim thresholds, labels couple through A and b.",
            good_for="Learning correlated label structure; BCE vs hinge-style losses (if added).",
            tags=(
                "deterministic-given-seed",
                "classification-style",
                "bce-friendly",
                "multi-label",
            ),
            children=(),
        )

    def _sample_inputs_rng_impl(self, **kwargs) -> np.ndarray:
        return self._rng_inputs.standard_normal((self.num_samples, self.d)).astype(float)

    def targets_from_inputs(self, X: np.ndarray) -> np.ndarray:
        x = self._as_validated_batch_inputs(X)
        logits = x @ self._A.T + self._b
        return (logits > 0).astype(float)
