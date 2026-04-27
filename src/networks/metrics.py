import numpy as np
from typing import Tuple


class Metrics:
    def __init__(self, metrics: dict = None):
        self.loss = metrics["loss"] if metrics else []
        self.regression_score = metrics["regression_score"] if metrics else []
        self.r_squared = metrics["r_squared"] if metrics else []
        self.adjusted_r_squared = metrics["adjusted_r_squared"] if metrics else []

        self.regression_score_sum = metrics["regression_score_sum"] if metrics else 0.0
        self.ss_res_sum = metrics["ss_res_sum"] if metrics else 0.0
        self.y_sum = metrics["y_sum"] if metrics else 0.0
        self.y2_sum = metrics["y2_sum"] if metrics else 0.0
        self.count = metrics["count"] if metrics else 0

    def __str__(self):
        loss = self.loss[-1] if self.loss else "N/A"
        regression_score = self.regression_score[-1] if self.regression_score else "N/A"
        r_squared = self.r_squared[-1] if self.r_squared else "N/A"
        adjusted_r_squared = self.adjusted_r_squared[-1] if self.adjusted_r_squared else "N/A"

        return (
            f"Loss: {loss}, Regression score: {regression_score}, "
            f"R^2: {r_squared}, Adjusted R^2: {adjusted_r_squared}"
        )

    def __repr__(self):
        return self.__str__()

    def as_dict(self):
        return {
            "loss": self.loss,
            "regression_score": self.regression_score,
            "r_squared": self.r_squared,
            "adjusted_r_squared": self.adjusted_r_squared,
            "regression_score_sum": self.regression_score_sum,
            "ss_res_sum": self.ss_res_sum,
            "y_sum": self.y_sum,
            "y2_sum": self.y2_sum,
            "count": self.count,
        }

    def add_metric(
        self, loss: float, regression_score: float, r_squared: float, adjusted_r_squared: float
    ):
        self.loss.append(loss)
        self.regression_score.append(regression_score)
        self.r_squared.append(r_squared)
        self.adjusted_r_squared.append(adjusted_r_squared)

    def tally_regression_score_r2(self, y_pred: np.ndarray, y_target: np.ndarray):
        diff_squared = (y_pred - y_target) ** 2
        rmse = np.sqrt(np.mean(diff_squared))
        score = np.exp(-rmse)
        self.regression_score_sum += score

        self.ss_res_sum += float(np.sum(diff_squared))
        self.y_sum += float(np.sum(y_target))
        self.y2_sum += float(np.sum(y_target**2))

        self.count += 1

    def calc_regression_score_and_r2(self, n: int, p: int = None) -> Tuple[float, float, float]:
        """
        Finalize epoch metrics: mean per-example exp(-RMSE), R², adjusted R².

        Args:
            n: Output dimensions per example (used in R² totals).
            p: Parameter count for adjusted R² (defaults to n if not provided).

        Returns:
            Tuple of (mean_regression_score, r_squared, adjusted_r_squared).
        """
        ss_tot = self.y2_sum - (self.y_sum**2 / (self.count * n))
        r_squared = 1 - (self.ss_res_sum / (ss_tot + 1e-12))

        if p is None:
            p = n
        num_samples = self.count * n
        if num_samples > p + 1:
            adjusted_r_squared = 1 - (1 - r_squared) * (num_samples - 1) / (num_samples - p - 1)
        else:
            adjusted_r_squared = r_squared

        return self.regression_score_sum / self.count, r_squared, adjusted_r_squared
