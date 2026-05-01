import numpy as np

from data.dataset import BaseDataset, DatasetNoiseMode
from hexnets_web.glossary_types import GlossaryNode


class SineDataset(BaseDataset, display_name="sine"):
    def __init__(
        self,
        d: int = 2,
        num_samples: int = 100,
        scale: float | np.float64 = 1.0,
        seed: int | None = None,
        omega: float = 1.0,
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
        self.omega = float(omega)
        self.data = None
        self.load_data()

    @classmethod
    def get_glossary_node(cls) -> GlossaryNode:
        return GlossaryNode(
            title="Sine dataset",
            aliases=("sine", "SineDataset"),
            english=(
                "Inputs **x** are uniform in [-1, 1]^d; targets **y = sin(ω x)** elementwise "
                "(default ω=1). CLI **scale** is accepted but not used."
            ),
            math_latex=r"y = \sin(\omega x)",
            example="Larger ω increases oscillation frequency and can make fitting harder.",
            good_for="Testing approximation capacity; under/overfitting behavior.",
            tags=(
                "deterministic-given-seed",
                "regression-compatible",
                "nonlinear",
                "periodic",
            ),
            children=(),
        )

    def _sample_inputs_rng_impl(self, **kwargs) -> np.ndarray:
        rng = np.random.default_rng(self.seed)
        return (rng.random((self.num_samples, self.d)) * 2 - 1).astype(float)

    def targets_from_inputs(self, X: np.ndarray) -> np.ndarray:
        x = self._as_validated_batch_inputs(X)
        return np.sin(self.omega * x)
