import os
from importlib import import_module

# Import the abstract base class
from services.figure_service.figure import Figure

# Import the service class
from services.figure_service.FigureService import FigureService

# Auto-import all figure classes (similar to activation/loss pattern)
for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith(".py") and file != "__init__.py" and file != "figure.py" and file != "FigureService.py":
        module = f"services.figure_service.{file[:-3]}"
        import_module(module)

# Import concrete figure classes
from services.figure_service.RefFigure import RefFigure
from services.figure_service.LearningRateRefFigure import LearningRateRefFigure
from services.figure_service.TrainingFigure import TrainingFigure

__all__ = ["Figure", "RefFigure", "LearningRateRefFigure", "TrainingFigure", "FigureService"]
