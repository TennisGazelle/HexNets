from abc import ABC, abstractmethod
import numpy as np


class BaseLoss(ABC):
    def __init__(self, name):
        self.name = name

    @abstractmethod
    def calc_loss(self, y_true, y_pred):
        pass

    @abstractmethod
    def calc_delta(self, y_true, y_pred):
        pass


class MeanSquaredError(BaseLoss):
    def __init__(self):
        super().__init__("mean_squared_error")

    def calc_loss(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    def calc_delta(self, y_true, y_pred):
        return (2 / y_true.shape[0]) * (y_pred - y_true)


class HuberLoss(BaseLoss):
    def __init__(self, delta_threshold=1.0):
        super().__init__(f"huber_loss_(delta-{delta_threshold})")
        self.delta_threshold = delta_threshold

    def calc_loss(self, y_true, y_pred):
        diff = y_pred - y_true
        abs_diff = np.abs(diff)
        quadratic = 0.5 * diff**2
        linear = self.delta_threshold * (abs_diff - 0.5 * self.delta_threshold)
        loss = np.where(abs_diff <= self.delta_threshold, quadratic, linear)
        # match MSE: return a scalar mean
        return np.mean(loss)

    def calc_delta(self, y_true, y_pred):
        diff = y_pred - y_true
        abs_diff = np.abs(diff)
        # elementwise grad (not normalized by N, to match your MSE.delta)
        return (
            np.where(
                abs_diff <= self.delta_threshold,
                diff,
                self.delta_threshold * np.sign(diff),
            )
            / y_true.shape[0]
        )


class LogCoshLoss(BaseLoss):
    def __init__(self):
        super().__init__("log_cosh_loss")

    def calc_loss(self, y_true, y_pred):
        diff = y_pred - y_true
        log_cosh = np.log(np.cosh(diff))
        return np.mean(log_cosh)

    def calc_delta(self, y_true, y_pred):
        diff = y_pred - y_true
        return np.tanh(diff)


class QuantileLoss(BaseLoss):
    def __init__(self, tau=0.5):
        super().__init__("quantile_loss")
        self.tau = tau

    def calc_loss(self, y_true, y_pred):
        diff = y_pred - y_true
        return np.mean(np.maximum(self.tau * diff, (self.tau - 1) * diff))

    def calc_delta(self, y_true, y_pred):
        diff = y_pred - y_true
        return np.where(diff >= 0, self.tau, self.tau - 1)
