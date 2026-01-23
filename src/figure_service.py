from abc import ABC, abstractmethod
import copy
import logging
import pathlib
import matplotlib.pyplot as plt
import numpy as np
from typing import Union
from logging_config import get_logger

logger = get_logger(__name__)


class Figure(ABC):
    @abstractmethod
    def __init__(self, filename: str):
        self.filename = filename

    @abstractmethod
    def save_figure(self):
        pass

    @abstractmethod
    def show_figure(self):
        pass

    @abstractmethod
    def update_figure(self, *args, **kwargs):
        pass


class RefFigure(Figure):
    def __init__(self, title: str, filename: str, detail: str):
        super().__init__(filename)
        self.title = title
        self.fig = plt.figure(figsize=(7, 7))
        self.fig.suptitle(self.title)
        self.fig.title(detail)

    def save_figure(self):
        self.fig.savefig()

    def show_figure(self):
        self.fig.show()

    def update_figure(self, *args, **kwargs):
        """Update the figure with new data. Placeholder implementation."""
        pass


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


class TrainingFigure(Figure):
    def __init__(self, title: str, filename: str, loss_detail: str, accuracy_detail: str, r2_detail: str):
        super().__init__(filename)
        self.title = title
        self.loss_detail = loss_detail
        self.accuracy_detail = accuracy_detail
        self.r2_detail = r2_detail

        self.channels = list(range(6))

        self.training_metrics = {channel: {"loss": [], "accuracy": [], "r_squared": []} for channel in self.channels}

        self.fig, (self.ax_loss, self.ax_acc, self.ax_r2) = plt.subplots(3, 1, figsize=(6, 12))
        self.fig.suptitle(f"{self.title}")

        self.lines_loss = {}
        self.lines_acc = {}
        self.lines_r2 = {}

        colors = plt.cm.tab10(np.linspace(0, 1, len(self.channels)))

        for channel in self.channels:
            (self.lines_loss[channel],) = self.ax_loss.plot([], [], label=f"Channel {channel}", color=colors[channel])
            (self.lines_acc[channel],) = self.ax_acc.plot([], [], label=f"Channel {channel}", color=colors[channel])
            (self.lines_r2[channel],) = self.ax_r2.plot([], [], label=f"Channel {channel}", color=colors[channel])

        self.ax_loss.legend()
        self.ax_loss.set_title(f"Loss ({self.loss_detail})")
        self.ax_loss.set_ylabel("Loss")
        self.ax_loss.grid(True)

        self.ax_acc.legend()
        self.ax_acc.set_title(f"Accuracy ({self.accuracy_detail})")
        self.ax_acc.set_ylabel("Accuracy")
        self.ax_acc.set_ylim(0, 1)
        self.ax_acc.grid(True)

        self.ax_r2.legend()
        self.ax_r2.set_title(f"R^2 ({self.r2_detail})")
        self.ax_r2.set_xlabel("Epoch")
        self.ax_r2.set_ylabel("R^2")
        self.ax_r2.grid(True)

    def save_figure(self):
        # Ensure filename is a Path object
        filename_path = pathlib.Path(self.filename) if isinstance(self.filename, str) else self.filename
        # Create parent directory if it doesn't exist
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

    def update_figure(self, training_metrics: dict, channel: int = 0):
        if any(not training_metrics[k] for k in ("loss", "accuracy", "r_squared")):
            return

        self.training_metrics[channel]["loss"].append(training_metrics["loss"])
        self.training_metrics[channel]["accuracy"].append(training_metrics["accuracy"])
        self.training_metrics[channel]["r_squared"].append(training_metrics["r_squared"])

        # loss
        self.lines_loss[channel].set_data(
            np.arange(1, len(self.training_metrics[channel]["loss"]) + 1),
            self.training_metrics[channel]["loss"],
        )
        self.ax_loss.relim()
        self.ax_loss.autoscale_view()

        # accuracy
        self.lines_acc[channel].set_data(
            np.arange(1, len(self.training_metrics[channel]["accuracy"]) + 1),
            self.training_metrics[channel]["accuracy"],
        )
        self.ax_acc.relim()
        self.ax_acc.autoscale_view()

        # r^2
        self.lines_r2[channel].set_data(
            np.arange(1, len(self.training_metrics[channel]["r_squared"]) + 1),
            self.training_metrics[channel]["r_squared"],
        )
        self.ax_r2.relim()
        self.ax_r2.autoscale_view()

        self.fig.canvas.draw()


class FigureService:
    def __init__(self):
        self.figures_path = pathlib.Path("figures")
        self.figures = {}

    def set_figures_path(self, figures_path: Union[pathlib.Path, None] = None):
        self.figures_path = pathlib.Path(figures_path) if figures_path else pathlib.Path("figures")

    def init_training_figure(self, filename, title, loss_detail, accuracy_detail, r2_detail):
        self.figures[title] = TrainingFigure(
            title, self.figures_path / filename, loss_detail, accuracy_detail, r2_detail
        )
        return self.figures[title]

    def init_ref_figure(self, filename, title, detail):
        self.figures[title] = RefFigure(title, filename, detail)
        return self.figures[title]

    def init_learning_rate_ref_figure(self, filename, title, learning_rate_name, max_iterations=500):
        self.figures[title] = LearningRateRefFigure(
            title, self.figures_path / filename, learning_rate_name, max_iterations
        )
        return self.figures[title]
