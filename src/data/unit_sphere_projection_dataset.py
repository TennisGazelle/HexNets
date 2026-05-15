import numpy as np

from data.dataset import GaussianInputProjectionDataset
from hexnets_web.glossary_types import GlossaryNode


class UnitSphereProjectionDataset(GaussianInputProjectionDataset, display_name="unit_sphere_projection"):
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

    def targets_from_inputs(self, X: np.ndarray) -> np.ndarray:
        x = self._as_validated_batch_inputs(X)
        norm = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
        return x / norm
