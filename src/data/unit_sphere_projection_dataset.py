import numpy as np

from data.dataset import BaseDataset, DatasetNoiseMode
from hexnets_web.glossary_types import GlossaryNode


class UnitSphereProjectionDataset(BaseDataset, display_name="unit_sphere_projection"):
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
        self.scale = float(scale)
        self.seed = seed
        self.data = None
        self.load_data()

    @classmethod
    def get_glossary_node(cls) -> GlossaryNode:
        return GlossaryNode(
            title="Unit sphere projection dataset",
            aliases=("unit_sphere_projection", "UnitSphereProjectionDataset"),
            english=(
                "Inputs **x** are standard normal; each row is projected to the **unit ℓ₂ sphere**: "
                "**y = x / ||x||₂** (per row). CLI **scale** is accepted but not used."
            ),
            math_latex=r"y = x / \|x\|_2",
            example="Every target row has Euclidean norm 1 (up to numerical error).",
            good_for="Direction learning; cosine loss; stability tests.",
            tags=(
                "deterministic-given-seed",
                "regression-compatible",
                "nonlinear",
                "projection",
                "normalized-outputs",
            ),
            children=(),
        )

    def _sample_inputs_rng_impl(self, **kwargs) -> np.ndarray:
        rng = np.random.default_rng(self.seed)
        return rng.standard_normal((self.num_samples, self.d)).astype(float)

    def targets_from_inputs(self, X: np.ndarray) -> np.ndarray:
        x = self._as_validated_batch_inputs(X)
        norm = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
        return x / norm
