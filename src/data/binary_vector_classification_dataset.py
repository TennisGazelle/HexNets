import numpy as np

from data.dataset import BaseDataset
from hexnets_web.glossary_types import GlossaryNode


class BinaryVectorClassificationDataset(BaseDataset, display_name="binary_vector_classification"):
    def __init__(
        self,
        d: int = 2,
        num_samples: int = 100,
        scale: float | np.float64 = 1.0,
        seed: int | None = None,
        threshold: float = 0.0,
    ):
        super().__init__()
        self.d = d
        self.num_samples = num_samples
        self.scale = float(scale)
        self.seed = seed
        self.threshold = float(threshold)
        self.data = None
        self.load_data()

    @classmethod
    def get_glossary_node(cls) -> GlossaryNode:
        return GlossaryNode(
            title="Binary vector classification dataset",
            aliases=("binary_vector_classification", "BinaryVectorClassificationDataset"),
            english=(
                "Inputs **x** are standard normal; targets are **0/1 floats** with **y_i = 1** iff **x_i > threshold** "
                "(default threshold 0). Intended for sigmoid + BCE-style training; MSE is still supported in-app. "
                "CLI **scale** is accepted but not used."
            ),
            math_latex=r"y_i = \mathbb{1}[x_i > t]",
            example="Per-dimension thresholding turns regression into a vector of Bernoulli labels.",
            good_for="Sigmoid output + BCE; calibration; thresholding tasks.",
            tags=(
                "deterministic-given-seed",
                "classification-style",
                "bce-friendly",
                "discrete-targets",
            ),
            children=(),
        )

    def load_data(self) -> bool:
        rng = np.random.default_rng(self.seed)
        X = rng.standard_normal((self.num_samples, self.d)).astype(float)
        Y = (X > self.threshold).astype(float)
        self.data = {"X": X, "Y": Y}
        return True
