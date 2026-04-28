import numpy as np

from data.dataset import BaseDataset, DatasetNoiseMode
from hexnets_web.glossary_types import GlossaryNode


class L2BallProjectionDataset(BaseDataset, display_name="l2_ball_projection"):
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
        self.r = float(scale)
        self.seed = seed
        self.data = None
        self.load_data()

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

    def _load_data_impl(self) -> None:
        rng = np.random.default_rng(self.seed)
        X = rng.standard_normal((self.num_samples, self.d)).astype(float)
        norm = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
        scale_row = np.minimum(1.0, self.r / norm)
        Y = X * scale_row
        self.data = {"X": X, "Y": Y}
