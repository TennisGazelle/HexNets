import numpy as np

from data.dataset import GaussianInputProjectionDataset
from hexnets_web.glossary_types import GlossaryNode


class NonNegativeProjectionDataset(GaussianInputProjectionDataset, display_name="non_negative_projection"):
    @classmethod
    def get_glossary_node(cls) -> GlossaryNode:
        return GlossaryNode(
            title="Non-negative projection dataset",
            aliases=("non_negative_projection", "NonNegativeProjectionDataset", "relu target"),
            english=(
                "Inputs **x** are standard normal; targets **y = max(x, 0)** elementwise (projection onto "
                "the nonnegative orthant). CLI **scale** is accepted but not used."
            ),
            math_latex=r"y = \max(x, 0)",
            example="Negative coordinates of X become 0 in Y; positive coordinates are unchanged.",
            good_for="Piecewise-linear learning; ReLU output experiments.",
            tags=(
                "deterministic-given-seed",
                "regression-compatible",
                "nonlinear",
                "projection",
                "piecewise-linear",
            ),
            children=(),
        )

    def targets_from_inputs(self, X: np.ndarray) -> np.ndarray:
        x = self._as_validated_batch_inputs(X)
        return np.maximum(x, 0.0)
