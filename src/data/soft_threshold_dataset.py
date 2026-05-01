import numpy as np

from data.dataset import BaseDataset, DatasetNoiseMode
from hexnets_web.glossary_types import GlossaryNode


class SoftThresholdDataset(BaseDataset, display_name="soft_threshold"):
    def __init__(
        self,
        d: int = 2,
        num_samples: int = 100,
        scale: float | np.float64 = 1.0,
        seed: int | None = None,
        lam: float = 0.25,
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
        self.lam = float(lam)
        self.data = None
        self.load_data()

    @classmethod
    def get_glossary_node(cls) -> GlossaryNode:
        return GlossaryNode(
            title="Soft-threshold dataset",
            aliases=("soft_threshold", "SoftThresholdDataset", "prox l1"),
            english=(
                "Inputs **x** are standard normal; **y = sign(x) max(|x| − λ, 0)** elementwise "
                "(soft threshold / prox of L1). Default λ=0.25. CLI **scale** is accepted but not used."
            ),
            math_latex=r"y = \operatorname{sign}(x)\,\max(|x|-\lambda,0)",
            example="Coordinates with |x| ≤ λ are driven to 0 in the target, encouraging sparse Y.",
            good_for="“Optimization operator” learning; sparsity behavior; robust losses.",
            tags=(
                "deterministic-given-seed",
                "regression-compatible",
                "nonlinear",
                "sparsity",
                "operator-learning",
            ),
            children=(),
        )

    def _sample_inputs_rng_impl(self, **kwargs) -> np.ndarray:
        rng = np.random.default_rng(self.seed)
        return rng.standard_normal((self.num_samples, self.d)).astype(float)

    def targets_from_inputs(self, X: np.ndarray) -> np.ndarray:
        x = self._as_validated_batch_inputs(X)
        return np.sign(x) * np.maximum(np.abs(x) - self.lam, 0.0)
