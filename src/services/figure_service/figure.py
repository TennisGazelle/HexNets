from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import numpy as np


class Figure(ABC):
    """Abstract base class for all figures."""

    colors = plt.cm.tab10(np.linspace(0, 1, 10))

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
