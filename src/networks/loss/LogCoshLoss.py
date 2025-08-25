import numpy as np

from src.networks.loss.loss import BaseLoss


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
