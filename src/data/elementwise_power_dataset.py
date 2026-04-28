import numpy as np

from data.dataset import BaseDataset
from hexnets_web.glossary_types import GlossaryNode


class ElementwisePowerDataset(BaseDataset, display_name="elementwise_power"):
    def __init__(
        self,
        d: int = 2,
        num_samples: int = 100,
        scale: float | np.float64 = 1.0,
        seed: int | None = None,
        p: float = 2.0,
    ):
        super().__init__()
        self.d = d
        self.num_samples = num_samples
        self.scale = float(scale)
        self.seed = seed
        self.p = float(p)
        self.data = None
        self.load_data()

    @classmethod
    def get_glossary_node(cls) -> GlossaryNode:
        return GlossaryNode(
            title="Elementwise power dataset",
            aliases=("elementwise_power", "ElementwisePowerDataset"),
            english=(
                "Inputs **x** are uniform in [-1, 1]^d; targets **y = sign(x) |x|^p** elementwise "
                "(parameter **p**, default 2). CLI **scale** is accepted but not used."
            ),
            math_latex=r"y = \operatorname{sign}(x)\,|x|^p",
            example="With p=2 and x=0.5, y=0.25; with x=-0.5, y=-0.25.",
            good_for="Nonlinearity learning; activation choice experiments.",
            tags=(
                "deterministic-given-seed",
                "regression-compatible",
                "nonlinear",
                "elementwise",
            ),
            children=(),
        )

    def load_data(self) -> bool:
        rng = np.random.default_rng(self.seed)
        X = (rng.random((self.num_samples, self.d)) * 2 - 1).astype(float)
        Y = np.sign(X) * (np.abs(X) ** self.p)
        self.data = {"X": X, "Y": Y}
        return True
