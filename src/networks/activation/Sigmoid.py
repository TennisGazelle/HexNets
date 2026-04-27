import numpy as np

from networks.activation.activations import BaseActivation
from streamlit_app.glossary_types import GlossaryNode


class Sigmoid(BaseActivation, display_name="sigmoid"):

    @classmethod
    def get_glossary_node(cls) -> GlossaryNode:
        return GlossaryNode(
            title="Sigmoid",
            aliases=("sigmoid", "logistic"),
            english=(
                "**activate**: 1/(1 + exp(−x)). **deactivate**: σ(x)(1 − σ(x)) using the same **activate** evaluation. "
                "Saturates for large |x|; gradients vanish in the tails."
            ),
            math_latex=r"\sigma(x)=\frac{1}{1+e^{-x}},\quad \sigma'=\sigma(1-\sigma)",
        )

    def activate(self, x):
        return 1 / (1 + np.exp(-x))

    def deactivate(self, x):
        return self.activate(x) * (1 - self.activate(x))
