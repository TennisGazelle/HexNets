import pathlib
import matplotlib.pyplot as plt
import numpy as np
from services.figure_service.figure import Figure
from services.logging_config import get_logger

logger = get_logger(__name__)


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

    def save_figure(self):
        filename_path = pathlib.Path(self.filename) if isinstance(self.filename, str) else self.filename
        filename_path.parent.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Saving figure to {filename_path.absolute()}")
        try:
            self.fig.savefig(filename_path)
            logger.debug(f"Successfully saved figure to {filename_path.absolute()}")
        except Exception as e:
            logger.error(f"Error saving figure: {e}")
            raise

    def show_figure(self):
        self.fig.show()

    def update_figure(self, iterations: np.ndarray, learning_rates: np.ndarray):
        self.line.set_data(iterations, learning_rates)
        self.ax.relim()
        self.ax.autoscale_view()
        self.fig.canvas.draw()
