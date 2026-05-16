from abc import ABC, abstractmethod
import pathlib
from typing import Any, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from networks.activation.activations import BaseActivation
from networks.loss.loss import BaseLoss
from networks.learning_rate.learning_rate import BaseLearningRate
from networks.activation.Sigmoid import Sigmoid
from networks.loss.MeanSquaredErrorLoss import MeanSquaredErrorLoss
from networks.learning_rate.ConstantLearningRate import ConstantLearningRate
from networks.learning_rate.learning_rate import get_learning_rate
from networks.metrics import Metrics
from services.logging_config import get_logger
from utils import table_print
from services.figure_service.FigureService import (
    FigureService,
    REGRESSION_SCORE_DETAIL,
    R2_DETAIL,
)

logger = get_logger(__name__)


class BaseNeuralNetwork(ABC):
    def __init__(
        self,
        learning_rate="constant",
        activation: BaseActivation = Sigmoid,
        loss: BaseLoss = MeanSquaredErrorLoss,
    ):
        # learning_rate can be a string (function name) or BaseLearningRate instance
        if isinstance(learning_rate, str):
            # Default to 0.01 for constant learning rate
            self.learning_rate_fn = get_learning_rate(learning_rate, learning_rate=0.01)
        elif isinstance(learning_rate, BaseLearningRate):
            self.learning_rate_fn = learning_rate
        else:
            # Backward compatibility: if float, use constant with that value
            self.learning_rate_fn = ConstantLearningRate(learning_rate=learning_rate)

        self.activation = activation
        self.loss = loss
        self.training_metrics = Metrics()
        self.epochs_completed = 0
        self.data_iteration = 0  # Track current data element index

    def __init_subclass__(self, **kwargs):
        self.display_name = kwargs.get("display_name", self.__class__.__name__.lower())

    def _init_figure_service(self, *, training_filename_prefix: str) -> None:

        self.figure_service = FigureService()
        self.figure_service.set_figures_path(None)
        self.training_figure = self.figure_service.init_training_figure(
            f"{training_filename_prefix}{self.loss}_{self.activation}.png",
            f"Training {self.display_name}",
            self.loss.display_name,
            REGRESSION_SCORE_DETAIL,
            R2_DETAIL,
        )

    @staticmethod
    def _latest_metrics_values(metrics: Metrics, epochs_completed: int) -> List[float]:
        if len(metrics.loss) == 0:
            return [0.0, 0.0, 0.0, 0.0, 0]
        return [
            metrics.loss[-1],
            metrics.regression_score[-1],
            metrics.r_squared[-1],
            metrics.adjusted_r_squared[-1] if metrics.adjusted_r_squared else 0.0,
            epochs_completed,
        ]

    @abstractmethod
    def _metrics_for_display(self) -> Metrics:
        pass

    @abstractmethod
    def _metrics_table_headers(self) -> List[str]:
        pass

    def _metrics_table_row_prefix(self) -> List[Any]:
        return []

    def show_latest_metrics(self) -> None:
        metrics = self._metrics_for_display()
        data = self._latest_metrics_values(metrics, self.epochs_completed)
        table_print(self._metrics_table_headers(), [[*self._metrics_table_row_prefix(), *data]])

    def _finalize_training_epoch(
        self,
        *,
        epoch_loss: float,
        epoch_reg_score: float,
        epoch_r2: float,
        epoch_adj_r2: float,
        training_channel: int,
        weights_figure,
        pause: float,
        show_training_metrics: bool,
        show_weights_live: bool,
        is_last_epoch: bool,
        weights_update_kwargs: Optional[dict] = None,
    ) -> None:
        self.training_figure.update_figure(
            {
                "loss": epoch_loss,
                "regression_score": epoch_reg_score,
                "r_squared": epoch_r2,
                "adjusted_r_squared": epoch_adj_r2,
            },
            training_channel,
            redraw=show_training_metrics,
        )
        self.apply_delta_W()

        if show_weights_live and weights_figure is not None:
            kwargs = weights_update_kwargs or {}
            weights_figure.update_figure(self.get_weight_matrices_for_live_plot(), **kwargs)

        if show_training_metrics or show_weights_live:
            plt.pause(pause)

        if is_last_epoch:
            logger.debug("About to save figure at end of training")
            self.training_figure.save_figure()
            logger.info("Training complete!")
            self.show_latest_metrics()
            logger.info(f"Training figure saved to: {self.training_figure.filename}")
            if weights_figure is not None:
                weights_figure.save_figure()
                logger.info(f"Weights figure saved to: {weights_figure.filename}")

    @abstractmethod
    def get_weight_matrices_for_live_plot(self):
        pass

    @abstractmethod
    def _training_metrics_r2_n(self) -> int:
        pass

    @abstractmethod
    def _training_metrics_r2_p(self) -> int:
        pass

    # def init_training_metrics(self):
    #     return {"loss": [], "regression_score": [], "r_squared": []}

    # @abstractmethod
    # def _init_from_file(self, filepath):
    #     pass

    @staticmethod
    @abstractmethod
    def get_parameter_count(self):
        raise NotImplementedError("get_parameter_count not implemented")

    @abstractmethod
    def show_stats(self):
        pass

    @abstractmethod
    def save(self, filepath) -> None:
        pass

    @abstractmethod
    def load(filepath) -> "BaseNeuralNetwork":
        pass

    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def backward(self, activations: np.ndarray, target: np.ndarray, apply_delta_W: bool = True):
        pass

    @abstractmethod
    def apply_delta_W(self):
        pass

    @abstractmethod
    def train(self, data, epochs: int = 1):
        pass

    @abstractmethod
    def train_animated(
        self,
        data,
        epochs: int = 1,
        pause: float = 0.05,
        output_dir: pathlib.Path = None,
        simple_figure_names: bool = False,
        show_training_metrics: bool = True,
        show_weights_live: bool = False,
    ):
        """
        Train while animating loss and regression score over epochs.

        - data: iterable of (x_input, y_target) with shapes (n,) and (n,)
        - epochs: number of epochs to train for
        - pause: time to pause between epochs
        - output_dir: directory to save figures to
        - simple_figure_names: use short stable names (training_metrics.png,
          weights_live.png) suited for run folders; default keeps descriptive
          names for standalone use.
        - show_training_metrics: live-update the metrics figure each epoch.
        - show_weights_live: open and live-update a weight heatmap each epoch.
        """
        pass

    @abstractmethod
    def test(self, x):
        pass

    @abstractmethod
    def graph_weights(self, activation_only=True, detail="", output_dir: pathlib.Path = None):
        pass

    @abstractmethod
    def graph_structure(self, detail="", output_dir: pathlib.Path = None) -> Tuple[str, plt.Figure]:
        pass

    @abstractmethod
    def get_metrics_json(self):
        pass
