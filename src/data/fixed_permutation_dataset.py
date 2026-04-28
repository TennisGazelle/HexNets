import numpy as np

from data.dataset import BaseDataset, DatasetNoiseMode
from hexnets_web.glossary_types import GlossaryNode


class FixedPermutationDataset(BaseDataset, display_name="fixed_permutation"):
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
        seq = np.random.SeedSequence(seed).spawn(2)
        self._rng_perm = np.random.default_rng(seq[0])
        self._rng_xy = np.random.default_rng(seq[1])
        self._perm = self._rng_perm.permutation(self.d)
        self.data = None
        self.load_data()

    @classmethod
    def get_glossary_node(cls) -> GlossaryNode:
        return GlossaryNode(
            title="Fixed permutation dataset",
            aliases=("fixed_permutation", "FixedPermutationDataset"),
            english=(
                "A **fixed** random permutation **P** of indices is chosen when the dataset is constructed "
                "(from **seed**). Inputs **x** are uniform in [-1,1]^d; targets **y = x[:, P]** (same values, "
                "shuffled coordinates). CLI **scale** is accepted but not used."
            ),
            math_latex=r"y = P x \quad \text{(coordinate permutation)}",
            example="If P swaps axes 0 and 1, the network must learn to unscramble coordinates.",
            good_for="Testing whether your architecture can learn index re-mapping.",
            tags=(
                "deterministic-given-seed",
                "regression-compatible",
                "linear",
                "permutation",
                "stress-test",
            ),
            children=(),
        )

    def _load_data_impl(self) -> None:
        X = (self._rng_xy.random((self.num_samples, self.d)) * 2 - 1).astype(float)
        Y = X[:, self._perm]
        self.data = {"X": X, "Y": Y}
