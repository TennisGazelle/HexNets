from networks.activation.activations import BaseActivation
from hexnets_web.glossary_types import GlossaryNode


class Linear(BaseActivation, display_name="linear"):
    @classmethod
    def get_glossary_node(cls) -> GlossaryNode:
        return GlossaryNode(
            title="Linear (identity) activation",
            aliases=("linear", "Linear", "identity activation"),
            english=(
                "**activate** returns **x** unchanged. **deactivate** returns the scalar **1.0** (derivative 1 everywhere). "
                "Useful when you want no nonlinearity on that layer path."
            ),
        )

    def activate(self, x):
        return x

    def deactivate(self, x):
        return 1.0
