import numpy as np
from typing import Tuple

class Metrics:
    def __init__(self, metrics: dict = None):
        self.loss = metrics["loss"] if metrics else []
        self.accuracy = metrics["accuracy"] if metrics else []
        self.r_squared = metrics["r_squared"] if metrics else []

        self.correct = metrics["correct"] if metrics else 0.0
        self.ss_res_sum = metrics["ss_res_sum"] if metrics else 0.0
        self.y_sum = metrics["y_sum"] if metrics else 0.0
        self.y2_sum = metrics["y2_sum"] if metrics else 0.0
        self.count = metrics["count"] if metrics else 0

    def __str__(self):
        loss = self.loss[-1] if self.loss else "N/A"
        accuracy = self.accuracy[-1] if self.accuracy else "N/A"
        r_squared = self.r_squared[-1] if self.r_squared else "N/A"

        return f"Loss: {loss}, Accuracy: {accuracy}, R^2: {r_squared}"

    def __repr__(self):
        return self.__str__()
    
    def as_dict(self):
        return {
            "loss": self.loss,
            "accuracy": self.accuracy,
            "r_squared": self.r_squared,
            "correct": self.correct,
            "ss_res_sum": self.ss_res_sum,
            "y_sum": self.y_sum,
            "y2_sum": self.y2_sum,
            "count": self.count,
        }
    
    def add_metric(self, loss: float, accuracy: float, r_squared: float):
        self.loss.append(loss)
        self.accuracy.append(accuracy)
        self.r_squared.append(r_squared)
    
    def tally_accurcy_r2(self, y_pred: np.ndarray, y_target: np.ndarray):
        diff_squared = (y_pred - y_target) ** 2
        rmse = np.sqrt(np.mean(diff_squared))
        score = np.exp(-rmse)
        self.correct += score

        self.ss_res_sum += float(np.sum(diff_squared))
        self.y_sum += float(np.sum(y_target))
        self.y2_sum += float(np.sum(y_target ** 2))

        self.count += 1

    def calc_accuracy_r2(self, n: int) -> Tuple[float, float]:
        ss_tot = self.y2_sum - (self.y_sum ** 2 / (self.count * n))
        r_squared = 1 - (self.ss_res_sum / (ss_tot + 1e-12))

        return self.correct / self.count, r_squared
