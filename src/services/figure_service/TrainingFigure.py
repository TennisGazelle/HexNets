import pathlib
import matplotlib.pyplot as plt
import numpy as np
from services.figure_service.figure import Figure
from services.logging_config import get_logger

logger = get_logger(__name__)


class TrainingFigure(Figure):
    def __init__(self, title: str, filename: str, loss_detail: str, regression_score_detail: str, r2_detail: str):
        super().__init__(filename)
        self.title = title
        self.loss_detail = loss_detail
        self.regression_score_detail = regression_score_detail
        self.r2_detail = r2_detail

        self.channels = list(range(6))

        self.training_metrics = {
            channel: {"loss": [], "regression_score": [], "r_squared": [], "adjusted_r_squared": []}
            for channel in self.channels
        }

        self.fig, (self.ax_loss, self.ax_reg_score, self.ax_r2, self.ax_adj_r2) = plt.subplots(4, 1, figsize=(6, 16))
        self.fig.suptitle(f"{self.title}")

        self.lines_loss = {}
        self.lines_reg_score = {}
        self.lines_r2 = {}
        self.lines_adj_r2 = {}

        colors = plt.cm.tab10(np.linspace(0, 1, len(self.channels)))

        for channel in self.channels:
            (self.lines_loss[channel],) = self.ax_loss.plot([], [], label=f"Channel {channel}", color=colors[channel])
            (self.lines_reg_score[channel],) = self.ax_reg_score.plot(
                [], [], label=f"Channel {channel}", color=colors[channel]
            )
            (self.lines_r2[channel],) = self.ax_r2.plot([], [], label=f"Channel {channel}", color=colors[channel])
            (self.lines_adj_r2[channel],) = self.ax_adj_r2.plot(
                [], [], label=f"Channel {channel}", color=colors[channel]
            )

        self.ax_loss.legend()
        self.ax_loss.set_title(f"Loss ({self.loss_detail})")
        self.ax_loss.set_ylabel("Loss")
        self.ax_loss.grid(True)

        self.ax_reg_score.legend()
        self.ax_reg_score.set_title(f"Regression score ({self.regression_score_detail})")
        self.ax_reg_score.set_ylabel("Mean exp(-RMSE)")
        self.ax_reg_score.set_ylim(0, 1)
        self.ax_reg_score.grid(True)

        self.ax_r2.legend()
        self.ax_r2.set_title(f"R^2 ({self.r2_detail})")
        self.ax_r2.set_ylabel("R^2")
        self.ax_r2.grid(True)

        self.ax_adj_r2.legend()
        self.ax_adj_r2.set_title(f"Adjusted R^2 ({self.r2_detail})")
        self.ax_adj_r2.set_xlabel("Epoch")
        self.ax_adj_r2.set_ylabel("Adjusted R^2")
        self.ax_adj_r2.grid(True)

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
        required_keys = ("loss", "regression_score", "r_squared")
        if not all(k in training_metrics for k in required_keys):
            return

        def _epoch_scalar_ok(x) -> bool:
            return isinstance(x, (int, float, np.integer, np.floating)) and not isinstance(x, bool)

        if not all(_epoch_scalar_ok(training_metrics[k]) for k in required_keys):
            return

        self.training_metrics[channel]["loss"].append(training_metrics["loss"])
        self.training_metrics[channel]["regression_score"].append(training_metrics["regression_score"])
        self.training_metrics[channel]["r_squared"].append(training_metrics["r_squared"])
        if "adjusted_r_squared" in training_metrics:
            self.training_metrics[channel]["adjusted_r_squared"].append(training_metrics["adjusted_r_squared"])

        # loss
        self.lines_loss[channel].set_data(
            np.arange(1, len(self.training_metrics[channel]["loss"]) + 1),
            self.training_metrics[channel]["loss"],
        )
        self.ax_loss.relim()
        self.ax_loss.autoscale_view()

        # regression score (mean per-example exp(-RMSE))
        self.lines_reg_score[channel].set_data(
            np.arange(1, len(self.training_metrics[channel]["regression_score"]) + 1),
            self.training_metrics[channel]["regression_score"],
        )
        self.ax_reg_score.relim()
        self.ax_reg_score.autoscale_view()

        # r^2
        self.lines_r2[channel].set_data(
            np.arange(1, len(self.training_metrics[channel]["r_squared"]) + 1),
            self.training_metrics[channel]["r_squared"],
        )
        self.ax_r2.relim()
        self.ax_r2.autoscale_view()

        # adjusted r^2
        self.lines_adj_r2[channel].set_data(
            np.arange(1, len(self.training_metrics[channel]["adjusted_r_squared"]) + 1),
            self.training_metrics[channel]["adjusted_r_squared"],
        )
        self.ax_adj_r2.relim()
        self.ax_adj_r2.autoscale_view()

        self.fig.canvas.draw()
