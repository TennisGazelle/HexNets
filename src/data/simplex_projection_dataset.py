import numpy as np

from data.dataset import BaseDataset, DatasetNoiseMode
from hexnets_web.glossary_types import GlossaryNode


class SimplexProjectionDataset(BaseDataset, display_name="simplex_projection"):
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

    def _sample_inputs_rng_impl(self, **kwargs) -> np.ndarray:
        rng = np.random.default_rng(self.seed)
        return rng.standard_normal((self.num_samples, self.d)).astype(float)

    def targets_from_inputs(self, X: np.ndarray) -> np.ndarray:
        x = self._as_validated_batch_inputs(X)
        return np.stack([self._proj_simplex(x[i]) for i in range(x.shape[0])], axis=0)
