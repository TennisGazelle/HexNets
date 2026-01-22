from networks.learning_rate.learning_rate import BaseLearningRate

class ConstantLearningRate(BaseLearningRate, display_name="constant"):
    def __init__(self, learning_rate: float = 0.01):
        self.learning_rate = learning_rate

    def rate_at_iteration(self, iteration: int) -> float:
        return self.learning_rate

