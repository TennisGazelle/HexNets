from abc import ABC, abstractmethod
from typing import ClassVar

import matplotlib.pyplot as plt
import numpy as np
import pathlib

from services.logging_config import get_logger

logger = get_logger(__name__)


class Figure(ABC):
    """Abstract base class for all figures."""

    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    save_log_label: ClassVar[str] = "figure"

    @abstractmethod
    def __init__(self, filename: str):
        self.filename = filename

    def _filename_path(self) -> pathlib.Path:
        path = pathlib.Path(self.filename) if isinstance(self.filename, str) else self.filename
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def save_figure(self) -> None:
        path = self._filename_path()
        logger.debug(f"Saving {self.save_log_label} to {path.absolute()}")
        try:
            self.fig.savefig(path)
            logger.debug(f"Successfully saved {self.save_log_label} to {path.absolute()}")
        except Exception as e:
            logger.error(f"Error saving {self.save_log_label}: {e}")
            raise

    def _canvas_draw(self, *, idle: bool = False) -> None:
        if idle:
            self.fig.canvas.draw_idle()
        else:
            self.fig.canvas.draw()

    def _refresh_line(self, ax, line, y, x=None) -> None:
        if x is None:
            x = np.arange(1, len(y) + 1)
        line.set_data(x, y)
        ax.relim()
        ax.autoscale_view()

    @abstractmethod
    def update_figure(self, *args, **kwargs):
        pass
