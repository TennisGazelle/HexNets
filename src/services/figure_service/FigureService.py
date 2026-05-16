import pathlib
from typing import Union, List, Tuple

import matplotlib.pyplot as plt

from services.figure_service.TrainingFigure import TrainingFigure
from services.figure_service.RefFigure import RefFigure
from services.figure_service.LearningRateRefFigure import LearningRateRefFigure
from services.figure_service.DynamicWeightsFigure import DynamicWeightsFigure

# Shared display strings used across all network types to avoid duplication.
REGRESSION_SCORE_DETAIL = "mean exp(-RMSE) per example"
R2_DETAIL = "coefficient of determination"

# Dict key prefix for live weight heatmaps in ``figures`` (stable across title/rotation updates).
WEIGHTS_LIVE_KEY_PREFIX = "weights_live"


class FigureService:
    def __init__(self):
        self.figures_path = pathlib.Path("figures")
        self.figures = {}

    @staticmethod
    def weights_live_figure_key(layer_shapes: List[Tuple[int, int]]) -> str:
        """Stable ``figures`` dict key from weight-matrix shapes (one live window per layout)."""
        flat = "__".join(f"{int(h)}x{int(w)}" for h, w in layer_shapes)
        return f"{WEIGHTS_LIVE_KEY_PREFIX}:{flat}"

    def set_figures_path(self, figures_path: Union[pathlib.Path, None] = None):
        self.figures_path = pathlib.Path(figures_path) if figures_path else pathlib.Path("figures")

    # --- training figure setup ---

    def init_training_figure(self, filename, title, loss_detail, regression_score_detail, r2_detail):
        self.figures[title] = TrainingFigure(
            title, self.figures_path / filename, loss_detail, regression_score_detail, r2_detail
        )
        return self.figures[title]

    def prepare_training_animation(
        self,
        training_figure: TrainingFigure,
        *,
        output_dir: Union[pathlib.Path, None],
        simple_names: bool,
        network_kind: str,
        display_name: str,
        loss,
        activation,
    ) -> None:
        """Configure *training_figure* for the upcoming training run.

        Sets the output path, canonical title, and the three metric-detail
        strings, then calls refresh_metadata() so the matplotlib figure
        reflects the new values immediately.  This replaces the duplicated
        15-line setup block that was copy-pasted into each network's
        train_animated().
        """
        self.set_figures_path(output_dir)

        if simple_names:
            basename = "training_metrics.png"
        else:
            existing = (
                pathlib.Path(training_figure.filename)
                if isinstance(training_figure.filename, str)
                else training_figure.filename
            )
            basename = existing.name

        training_figure.filename = self.figures_path / basename
        training_figure.title = (
            f"{network_kind} Network Training {display_name}" f" (loss={loss}, activation={activation.display_name})"
        )
        training_figure.loss_detail = loss.display_name
        training_figure.regression_score_detail = REGRESSION_SCORE_DETAIL
        training_figure.r2_detail = R2_DETAIL
        training_figure.refresh_metadata()

    # --- weights live figure ---

    def _close_weights_live_figures_except(self, keep_key: str) -> None:
        prefix = f"{WEIGHTS_LIVE_KEY_PREFIX}:"
        for k in list(self.figures.keys()):
            if not isinstance(k, str) or not k.startswith(prefix) or k == keep_key:
                continue
            wf = self.figures.pop(k)
            if isinstance(wf, DynamicWeightsFigure):
                plt.close(wf.fig)

    def init_weights_live_figure(
        self,
        filename: str,
        title: str,
        layer_shapes: List[Tuple[int, int]],
    ):
        """Create or reuse a DynamicWeightsFigure in ``self.figures`` under a shape-derived key.

        Same layout (``layer_shapes``) reuses the same matplotlib window so rotation /
        successive ``train_animated`` calls do not spawn duplicate heatmaps.  A different
        layout closes other ``weights_live:…`` entries and adds a new figure under the new key.
        """
        key = self.weights_live_figure_key(layer_shapes)
        path = self.figures_path / filename
        self._close_weights_live_figures_except(key)
        existing = self.figures.get(key)
        if existing is not None and isinstance(existing, DynamicWeightsFigure):
            existing.filename = path
            existing.title = title
            existing.fig.suptitle(title)
            return existing

        wf = DynamicWeightsFigure(title, path, layer_shapes)
        self.figures[key] = wf
        return wf

    # --- reference figures ---

    def init_ref_figure(self, filename, title, detail):
        self.figures[title] = RefFigure(title, filename, detail)
        return self.figures[title]

    def init_learning_rate_ref_figure(self, filename, title, learning_rate_name, max_iterations=500):
        self.figures[title] = LearningRateRefFigure(
            title, self.figures_path / filename, learning_rate_name, max_iterations
        )
        return self.figures[title]
