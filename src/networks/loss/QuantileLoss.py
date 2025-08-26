import numpy as np

from src.networks.loss.loss import BaseLoss


class QuantileLoss(BaseLoss, display_name="quantile"):
    def __init__(self, tau=0.5):
        self.tau = tau

    def calc_loss(self, y_true, y_pred):
        diff = y_pred - y_true
        return np.mean(np.maximum(self.tau * diff, (self.tau - 1) * diff))

    def calc_delta(self, y_true, y_pred):
        diff = y_pred - y_true
        return np.where(diff >= 0, self.tau, self.tau - 1)
