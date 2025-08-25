import numpy as np

from src.networks.loss.loss import BaseLoss


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
