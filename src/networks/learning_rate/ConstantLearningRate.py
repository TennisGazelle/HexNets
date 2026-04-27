from networks.learning_rate.learning_rate import BaseLearningRate
from streamlit_app.glossary_types import GlossaryNode


class ConstantLearningRate(BaseLearningRate, display_name="constant"):
    @classmethod
    def get_glossary_node(cls) -> GlossaryNode:
        return GlossaryNode(
            title="Constant learning rate",
            aliases=("constant", "ConstantLearningRate", "fixed lr"),
            english=(
                "**rate_at_iteration** returns the same scalar for every iteration. "
                "Constructor argument **learning_rate** (default 0.01) is that fixed step size."
            ),
            example="With learning_rate=0.01, the effective multiplier is 0.01 on every update.",
        )

    def __init__(self, learning_rate: float = 0.01):
        self.learning_rate = learning_rate

    def rate_at_iteration(self, iteration: int) -> float:
        return self.learning_rate
