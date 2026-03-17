from abc import ABC, abstractmethod


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
