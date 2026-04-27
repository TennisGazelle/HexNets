import pathlib
from typing import Union
from services.figure_service.TrainingFigure import TrainingFigure
from services.figure_service.RefFigure import RefFigure
from services.figure_service.LearningRateRefFigure import LearningRateRefFigure


class FigureService:
    def __init__(self):
        self.figures_path = pathlib.Path("figures")
        self.figures = {}

    def set_figures_path(self, figures_path: Union[pathlib.Path, None] = None):
        self.figures_path = pathlib.Path(figures_path) if figures_path else pathlib.Path("figures")

    def init_training_figure(self, filename, title, loss_detail, regression_score_detail, r2_detail):
        self.figures[title] = TrainingFigure(
            title, self.figures_path / filename, loss_detail, regression_score_detail, r2_detail
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
