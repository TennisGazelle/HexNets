from abc import ABC, abstractmethod
import pathlib
import matplotlib.pyplot as plt
import numpy as np
from typing import Union

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
        self.fig = plt.figure(figsize=(7,7))
        self.fig.suptitle(self.title)
        self.fig.title(detail)
    
    def save_figure(self):
        self.fig.savefig()
    
    def show_figure(self):
        self.fig.show()


class TrainingFigure(Figure):
    def __init__(self, title: str, filename: str, loss_detail: str, accuracy_detail: str, r2_detail: str, training_metrics: dict):
        super().__init__(filename)
        self.title = title
        self.loss_detail = loss_detail
        self.accuracy_detail = accuracy_detail
        self.r2_detail = r2_detail
        self.training_metrics = training_metrics

        self.fig, (self.ax_loss, self.ax_acc, self.ax_r2) = plt.subplots(3, 1, figsize=(6, 12))
        self.fig.suptitle(f"{self.title}")

        (self.line_loss,) = self.ax_loss.plot([], [])
        self.ax_loss.set_title(f"Loss ({self.loss_detail})")
        self.ax_loss.set_ylabel("Loss")
        self.ax_loss.grid(True)

        (self.line_acc,) = self.ax_acc.plot([], [])
        self.ax_acc.set_title(f"Accuracy ({self.accuracy_detail})")
        self.ax_acc.set_ylabel("Accuracy")
        self.ax_acc.set_ylim(0, 1)
        self.ax_acc.grid(True)

        (self.line_r2,) = self.ax_r2.plot([], [])
        self.ax_r2.set_title(f"R^2 ({self.r2_detail})")
        self.ax_r2.set_xlabel("Epoch")
        self.ax_r2.set_ylabel("R^2")
        self.ax_r2.grid(True)

    def save_figure(self):
        self.fig.savefig(self.filename)

    def show_figure(self):
        self.fig.show()

    def update_figure(self, *args, **kwargs):
        self.training_metrics["loss"].append(kwargs["loss"])
        self.training_metrics["accuracy"].append(kwargs["accuracy"])
        self.training_metrics["r_squared"].append(kwargs["r_squared"])

        self.line_loss.set_data(np.arange(1, len(self.training_metrics["loss"]) + 1), self.training_metrics["loss"])
        self.ax_loss.relim()
        self.ax_loss.autoscale_view()

        self.fig.canvas.draw()
        self.line_acc.set_data(
            np.arange(1, len(self.training_metrics["accuracy"]) + 1), self.training_metrics["accuracy"]
        )
        self.ax_acc.relim()
        self.ax_acc.autoscale_view()

        self.fig.canvas.draw()
        self.line_r2.set_data(
            np.arange(1, len(self.training_metrics["r_squared"]) + 1), self.training_metrics["r_squared"]
        )
        self.ax_r2.relim()
        self.ax_r2.autoscale_view()

        self.fig.canvas.draw()


class FigureService:
    def __init__(self):
        self.figures_path = pathlib.Path("figures")
        self.figures = {}

    def set_figures_path(self, figures_path: Union[pathlib.Path, None]):
        self.figures_path = figures_path if figures_path else pathlib.Path("figures")

    def init_training_figure(self, filename, title, loss_detail, accuracy_detail, r2_detail, training_metrics):
        self.figures[title] = TrainingFigure(
            title, self.figures_path / filename, loss_detail, accuracy_detail, r2_detail, training_metrics
        )
        return self.figures[title]

    def init_ref_figure(self, filename, title, detail):
        self.figures[title] = RefFigure(
            title, filename, detail
        )
        return self.figures[title]