import numpy as np

from networks.activation.activations import BaseActivation
from streamlit_app.glossary_types import GlossaryNode


class ReLU(BaseActivation, display_name="relu"):

    @classmethod
    def get_glossary_node(cls) -> GlossaryNode:
        return GlossaryNode(
            title="ReLU",
            aliases=("relu", "ReLU", "rectified linear"),
            english=(
                "**activate**: max(0, x). **deactivate**: 1 where x > 0, else 0 (float mask). "
                "Standard piecewise-linear nonlinearity for hidden units."
            ),
            math_latex=r"\sigma(x)=\max(0,x),\quad \sigma'(x)=\mathbf{1}_{x>0}",
        )

    def activate(self, x):
        return np.maximum(0, x)

    def deactivate(self, x):
        return (x > 0).astype(float)
