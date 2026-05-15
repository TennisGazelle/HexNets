import numpy as np

from data.dataset import GaussianInputProjectionDataset
from hexnets_web.glossary_types import GlossaryNode


class SimplexProjectionDataset(GaussianInputProjectionDataset, display_name="simplex_projection"):
    @staticmethod
    def _proj_simplex(v: np.ndarray) -> np.ndarray:
        """Euclidean projection of vector v onto the probability simplex."""
        n = v.size
        u = np.sort(v)[::-1]
        cssv = np.cumsum(u)
        idx = np.arange(1, n + 1, dtype=float)
        cond = u * idx > (cssv - 1)
        rho = int(np.nonzero(cond)[0][-1])
        theta = (cssv[rho] - 1.0) / (rho + 1)
        return np.maximum(v - theta, 0.0)

    @classmethod
    def get_glossary_node(cls) -> GlossaryNode:
        return GlossaryNode(
            title="Simplex projection dataset",
            aliases=("simplex_projection", "SimplexProjectionDataset"),
            english=(
                "Inputs **x** are standard normal; each row is projected onto the **probability simplex** "
                "(nonnegative entries summing to 1). CLI **scale** is accepted but not used."
            ),
            math_latex=r"y = \Pi_\Delta(x)",
            example="Each row of Y is a probability vector suitable for KL-style objectives.",
            good_for="“Distribution” outputs; KL-divergence losses; constrained learning.",
            tags=(
                "deterministic-given-seed",
                "regression-compatible",
                "nonlinear",
                "projection",
                "probability-simplex",
            ),
            children=(),
        )

    def targets_from_inputs(self, X: np.ndarray) -> np.ndarray:
        x = self._as_validated_batch_inputs(X)
        return np.stack([self._proj_simplex(x[i]) for i in range(x.shape[0])], axis=0)
