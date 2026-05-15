import numpy as np

from data.dataset import GaussianInputProjectionDataset
from hexnets_web.glossary_types import GlossaryNode


class L2BallProjectionDataset(GaussianInputProjectionDataset, display_name="l2_ball_projection"):
    @classmethod
    def get_glossary_node(cls) -> GlossaryNode:
        return GlossaryNode(
            title="L2 ball projection dataset",
            aliases=("l2_ball_projection", "L2BallProjectionDataset"),
            english=(
                "Inputs **x** are standard normal; each row is projected onto the **closed ℓ₂ ball** "
                "of radius **r** = CLI **scale** (default 1): if ||x|| ≤ r then y=x, else y = r x/||x||."
            ),
            math_latex=r"y = \begin{cases}x & \|x\|\le r\\ r\,x/\|x\| & \text{otherwise}\end{cases}",
            example="With scale=r=1, vectors longer than 1 are scaled down to norm 1.",
            good_for="Bounded outputs; robust training; “prox-like” learning.",
            tags=(
                "deterministic-given-seed",
                "regression-compatible",
                "nonlinear",
                "projection",
                "bounded-outputs",
            ),
            children=(),
        )

    def targets_from_inputs(self, X: np.ndarray) -> np.ndarray:
        x = self._as_validated_batch_inputs(X)
        norm = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
        radius = float(self.scale)
        scale_row = np.minimum(1.0, radius / norm)
        return x * scale_row
