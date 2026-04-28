import numpy as np

from networks.activation.activations import BaseActivation
from hexnets_web.glossary_types import GlossaryNode


class LeakyReLU(BaseActivation, display_name="leaky_relu"):
    @classmethod
    def get_glossary_node(cls) -> GlossaryNode:
        return GlossaryNode(
            title="Leaky ReLU",
            aliases=("leaky_relu", "LeakyReLU", "LReLU"),
            english=(
                "**activate**: x where x > 0, else **alpha · x**. **deactivate**: 1 where x > 0, else **alpha**. "
                "Constructor **alpha** defaults to 0.01 so negative inputs still get a small gradient."
            ),
            math_latex=r"\sigma(x)=\max(\alpha x,x)",
        )

    def __init__(self, alpha=0.01):
        self.alpha = alpha

    def activate(self, x):
        return np.where(x > 0, x, self.alpha * x)

    def deactivate(self, x):
        return np.where(x > 0, 1, self.alpha)
