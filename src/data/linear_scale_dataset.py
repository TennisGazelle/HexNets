import numpy as np

from data.dataset import BaseDataset, DatasetNoiseMode
from hexnets_web.glossary_types import GlossaryNode


class LinearScaleDataset(BaseDataset, display_name="linear_scale"):
    def __init__(
        self,
        d: int = 2,
        num_samples: int = 100,
        scale: float | np.float64 = 1.0,
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
        self.scale = scale
        self.data = None

        self.load_data()

    @classmethod
    def get_glossary_node(cls) -> GlossaryNode:
        return GlossaryNode(
            title="Linear (scaled) dataset",
            aliases=(
                "linear_scale",
                "type=linear_scale",
                "LinearScaleDataset",
                "linear",
                "type=linear",
            ),
            english=(
                "Inputs **x** are drawn uniformly in [-1, 1]^d; targets are **y = scale · x** with a "
                "scalar `scale` (default 2.0 in `hexnet train` for this dataset). CLI and manifests use the id "
                "**linear_scale** (same as the dataset `display_name`). The Streamlit **Train Network** "
                "button uses the identity variant (`get_dataset(..., type='identity')`), not this one."
            ),
            math_latex=r"y = s \cdot x",
            example="With scale=2, if x = [0.5, -1.0] then y = [1.0, -2.0].",
            good_for="Checking optimizer sensitivity to magnitude; learning-rate tuning.",
            tags=(
                "deterministic",
                "regression-compatible",
                "linear",
                "affine-operator",
            ),
            children=(),
        )

    def _sample_inputs_rng_impl(self, **kwargs) -> np.ndarray:
        return (np.random.rand(self.num_samples, self.d) * 2 - 1).astype(float)

    def targets_from_inputs(self, X: np.ndarray) -> np.ndarray:
        x = self._as_validated_batch_inputs(X)
        return x.copy() * float(self.scale)
