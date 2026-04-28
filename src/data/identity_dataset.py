import numpy as np

from data.linear_scale_dataset import LinearScaleDataset
from hexnets_web.glossary_types import GlossaryNode


class IdentityDataset(LinearScaleDataset, display_name="identity"):
    def __init__(
        self,
        d: int = 2,
        num_samples: int = 100,
        scale: float | np.float64 | None = None,
    ):
        super().__init__(d, num_samples)
        self.scale = 1.0

    @classmethod
    def get_glossary_node(cls) -> GlossaryNode:
        return GlossaryNode(
            title="Identity dataset",
            aliases=("identity", "type=identity"),
            english=(
                "Training pairs where each target **y** equals the corresponding input **x**. "
                "In code this is `IdentityDataset`: it uses the same random **x** in [-1, 1]^d as "
                "`LinearScaleDataset` but with scale 1, so **y = x**."
            ),
            math_latex=r"y = x \quad \text{(elementwise)}",
            example="For d=3, one sample might be x = [0.2, -0.5, 0.1] and y = [0.2, -0.5, 0.1].",
            good_for="Sanity checks; debugging shapes; baseline convergence.",
            tags=(
                "sanity-check",
                "deterministic",
                "regression-compatible",
                "linear",
                "affine-operator",
            ),
            children=(),
        )
