from networks.learning_rate.learning_rate import BaseLearningRate


class ExponentialDecayLearningRate(BaseLearningRate, display_name="exponential_decay"):
    def __init__(self, initial_learning_rate: float = 0.01, decay_rate: float = 0.95, learning_rate: float = None):
        # Accept 'learning_rate' for compatibility with existing code
        if learning_rate is not None:
            self.initial_learning_rate = learning_rate
        else:
            self.initial_learning_rate = initial_learning_rate
        self.decay_rate = decay_rate

    def rate_at_iteration(self, iteration: int) -> float:
        return self.initial_learning_rate * (self.decay_rate ** iteration)
