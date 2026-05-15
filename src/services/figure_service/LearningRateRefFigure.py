import matplotlib.pyplot as plt
import numpy as np
from services.figure_service.figure import Figure


class LearningRateRefFigure(Figure):
    def __init__(self, title: str, filename: str, learning_rate_name: str, max_iterations: int = 500):
        super().__init__(filename)
        self.title = title
        self.learning_rate_name = learning_rate_name
        self.max_iterations = max_iterations

        self.fig, self.ax = plt.subplots(1, 1, figsize=(8, 6))
        self.fig.suptitle(f"{self.title}")

        self.ax.set_xlabel("Iteration")
        self.ax.set_ylabel("Learning Rate")
        self.ax.set_title(f"Learning Rate: {self.learning_rate_name}")
        self.ax.grid(True)
        self.ax.set_xlim(0, max_iterations)

        (self.line,) = self.ax.plot([], [], label=f"LR: {self.learning_rate_name}")
        self.ax.legend()

    def update_figure(self, iterations: np.ndarray, learning_rates: np.ndarray):
        self._refresh_line(self.ax, self.line, learning_rates, x=iterations)
        self._canvas_draw()
