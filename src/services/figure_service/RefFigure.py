import matplotlib.pyplot as plt
from services.figure_service.figure import Figure


class RefFigure(Figure):
    def __init__(self, title: str, filename: str, detail: str):
        super().__init__(filename)
        self.title = title
        self.fig = plt.figure(figsize=(7, 7))
        self.fig.suptitle(self.title)
        ax = self.fig.add_subplot(111)
        ax.set_title(detail)

    def save_figure(self):
        self.fig.savefig()

    def show_figure(self):
        self.fig.show()

    def update_figure(self, *args, **kwargs):
        """Update the figure with new data. Placeholder implementation."""
        pass
