from networks.learning_rate.learning_rate import BaseLearningRate
from hexnets_web.glossary_types import GlossaryNode


class ExponentialDecayLearningRate(BaseLearningRate, display_name="exponential_decay"):
    @classmethod
    def get_glossary_node(cls) -> GlossaryNode:
        return GlossaryNode(
            title="Exponential decay learning rate",
            aliases=("exponential_decay", "ExponentialDecayLearningRate", "exp decay"),
            english=(
                "**rate_at_iteration(iteration)** returns **initial_learning_rate · decay_rate^iteration**. "
                "Defaults: **initial_learning_rate** 0.01, **decay_rate** 0.95. "
                "If keyword **learning_rate** is passed, it is used as **initial_learning_rate** for compatibility with older call sites."
            ),
            math_latex=r"\eta_t = \eta_0 \cdot \gamma^t",
            example="With η₀=0.01 and γ=0.95, iteration 10 gives η≈0.00599.",
        )

    def __init__(self, initial_learning_rate: float = 0.01, decay_rate: float = 0.95, learning_rate: float = None):
        # Accept 'learning_rate' for compatibility with existing code
        if learning_rate is not None:
            self.initial_learning_rate = learning_rate
        else:
            self.initial_learning_rate = initial_learning_rate
        self.decay_rate = decay_rate

    def rate_at_iteration(self, iteration: int) -> float:
        return self.initial_learning_rate * (self.decay_rate**iteration)
