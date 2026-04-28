import numpy as np

from data.dataset import DatasetNoiseMode
from data.linear_scale_dataset import LinearScaleDataset
from hexnets_web.glossary_types import GlossaryNode


class DiagonalScaleDataset(LinearScaleDataset, display_name="diagonal_scale"):
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
            d,
            num_samples,
            scale,
            noise_mode=noise_mode,
            noise_mu=noise_mu,
            noise_sigma=noise_sigma,
            noise_seed=noise_seed,
        )
        self.data = None
        self.load_data()

    @classmethod
    def get_glossary_node(cls) -> GlossaryNode:
        return GlossaryNode(
            title="Diagonal scale dataset",
            aliases=("diagonal_scale", "type=diagonal_scale", "DiagonalScaleDataset"),
            english=(
                "Like the linear scaled dataset, **x** is uniform in [-1, 1]^d, but each output "
                "coordinate is scaled independently: **y_i = s · (i+1) · x_i** for zero-based index **i**. "
                "Useful for per-dimension behavior and conditioning; CLI id **diagonal_scale**."
            ),
            math_latex=r"y_i = s \cdot (i+1) \cdot x_i",
            example=(
                "With d=3 and scale=1, if x = [0.2, 0.5, -0.1] then "
                "y = [0.2, 1.0, -0.3] (dimension i is scaled by (i+1)·s)."
            ),
            good_for=(
                "Per-dimension learning behavior; fixed (non-random) diagonal conditioning; "
                "Adam vs SGD comparisons. (For random diagonal gains per dimension, see **diagonal_linear**.)"
            ),
            tags=(
                "deterministic",
                "regression-compatible",
                "linear",
                "per-dimension-conditioning",
            ),
            children=(),
        )

    def _load_data_impl(self) -> None:
        X = (np.random.rand(self.num_samples, self.d) * 2 - 1).astype(float)
        Y = X.copy()
        for i in range(self.d):
            Y[:, i] *= (i + 1) * self.scale
        self.data = {
            "X": X,
            "Y": Y,
        }
