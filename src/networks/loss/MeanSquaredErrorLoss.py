import numpy as np

from src.networks.loss.loss import BaseLoss


class MeanSquaredErrorLoss(BaseLoss, display_name="mean_squared_error"):

    def calc_loss(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    def calc_delta(self, y_true, y_pred):
        return (2 / y_true.shape[0]) * (y_pred - y_true)
