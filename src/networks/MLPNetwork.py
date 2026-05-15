import pickle
from typing import Any, List, Mapping, Union, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pathlib
import json
from tqdm import trange

from services.figure_service.FigureService import FigureService
from services.figure_service.weight_heatmap import render_weight_panels
from services.figure_service.DynamicWeightsFigure import DynamicWeightsFigure
from networks.network import BaseNeuralNetwork
from networks.activation.activations import BaseActivation
from networks.loss.loss import BaseLoss
from networks.activation.Sigmoid import Sigmoid
from networks.loss.MeanSquaredErrorLoss import MeanSquaredErrorLoss
from networks.activation.activations import get_activation_function
from networks.loss.loss import get_loss_function
from networks.metrics import Metrics
from data.dataset import BaseDataset
from utils import table_print
from services.logging_config import get_logger

logger = get_logger(__name__)


class MLPNetwork(BaseNeuralNetwork, display_name="mlp"):
    def __init__(
        self,
        input_dim: int = 3,
        output_dim: int = 3,
        hidden_dims: List[int] = [3],
        learning_rate: float = 0.01,
        activation: BaseActivation = Sigmoid,
        loss: BaseLoss = MeanSquaredErrorLoss,
    ):
        super().__init__(learning_rate, activation, loss)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.W = []
        self.delta_W = []
        self._init_figure_service(training_filename_prefix="mlpnet_training_")

        if len(hidden_dims) == 0:
            raise ValueError("hidden_dims must be a list of at least one integer")

        # add layers
        self._add_layer(input_dim, hidden_dims[0])
        for i in range(1, len(hidden_dims)):
            self._add_layer(hidden_dims[i - 1], hidden_dims[i])
        self._add_layer(hidden_dims[-1], output_dim)

    # --- static methods ---
    @staticmethod
    def get_run_metadata_schema() -> Mapping[str, Any]:
        """Describe ``model_metadata`` keys persisted in run ``config.json`` for ``model_type`` mlp."""
        return {
            "model_type": "mlp",
            "keys": {
                "input_dim": {"required": True, "type": "int", "min": 1},
                "output_dim": {"required": True, "type": "int", "min": 1},
                "hidden_dims": {"required": True, "type": "list[int]"},
            },
            "train_constraints": ("input_dim must equal output_dim for CLI/train template parity",),
        }

    @staticmethod
    def validate_run_metadata(meta: dict, *, require_square_io_for_train: bool = False) -> None:
        """
        Validate ``model_metadata`` from a run or train ``config.json``.

        When ``require_square_io_for_train`` is True, enforce ``input_dim == output_dim`` (matches CLI ``--num_dims``).
        """
        if not isinstance(meta, dict):
            raise ValueError("Run config: mlp model_metadata must be an object.")
        for k in ("input_dim", "output_dim", "hidden_dims"):
            if k not in meta:
                raise ValueError(f"Run config: mlp model_metadata requires '{k}'.")
        try:
            input_dim = int(meta["input_dim"])
            output_dim = int(meta["output_dim"])
        except (TypeError, ValueError) as e:
            raise ValueError("Run config: mlp model_metadata input_dim and output_dim must be integers.") from e
        hd = meta["hidden_dims"]
        if not isinstance(hd, list) or len(hd) == 0:
            raise ValueError("Run config: mlp model_metadata hidden_dims must be a non-empty list.")
        try:
            hidden_dims = [int(x) for x in hd]
        except (TypeError, ValueError) as e:
            raise ValueError("Run config: mlp model_metadata hidden_dims must be a list of integers.") from e
        if len(hidden_dims) == 0:
            raise ValueError("Run config: mlp model_metadata hidden_dims must be a non-empty list.")
        if require_square_io_for_train and input_dim != output_dim:
            raise ValueError("Run config: MLP training expects input_dim == output_dim (matches CLI --num_dims).")

    @staticmethod
    def get_parameter_count(input_dim: int, output_dim: int, hidden_dims: List[int]) -> int:
        if len(hidden_dims) == 0:
            raise ValueError("hidden_dims must be a list of at least one integer")
        dims = [input_dim] + hidden_dims + [output_dim]
        return sum(dims[i] * dims[i + 1] for i in range(len(dims) - 1))

    # --- instance methods ---
    def _metrics_for_display(self) -> Metrics:
        return self.training_metrics

    def _metrics_table_headers(self) -> List[str]:
        return ["Loss", "Reg. score", "R^2", "Adjusted R^2", "Epochs"]

    def _training_metrics_r2_n(self) -> int:
        return self.output_dim

    def _training_metrics_r2_p(self) -> int:
        return self.input_dim

    def get_weight_matrices_for_live_plot(self):
        """Return the current weight matrices for live heatmap display."""
        return self.W

    # --- structure helpers ---
    def _add_layer(self, incoming_dim: int, outgoing_dim: int):
        weights = np.random.random((incoming_dim, outgoing_dim)) - 0.5
        self.W.append(weights)
        self.delta_W.append(np.zeros((incoming_dim, outgoing_dim)))

    def save(self, filepath) -> None:
        with open(filepath, "wb") as f:
            pickle.dump(
                {
                    "input_dim": self.input_dim,
                    "output_dim": self.output_dim,
                    "hidden_dims": self.hidden_dims,
                    "weights": self.W,
                    "training_metrics": self.training_metrics.as_dict(),
                    "learning_rate": self.learning_rate_fn.display_name,
                    "activation": self.activation.display_name,
                    "loss": self.loss.display_name,
                    "epochs_completed": self.epochs_completed,
                    "data_iteration": self.data_iteration,
                },
                f,
            )

    def load(self, filepath):
        with open(filepath, "rb") as f:
            data = pickle.load(f)
            self.input_dim = data["input_dim"]
            self.output_dim = data["output_dim"]
            self.hidden_dims = data["hidden_dims"]
            self.W = data["weights"]
            self.training_metrics = Metrics(data["training_metrics"])
            self.activation = get_activation_function(data["activation"])
            self.loss = get_loss_function(data["loss"])
            # Handle backward compatibility: if learning_rate is a float, use constant
            learning_rate_config = data.get("learning_rate", "constant")
            if isinstance(learning_rate_config, (int, float)):
                from networks.learning_rate.ConstantLearningRate import (
                    ConstantLearningRate,
                )

                self.learning_rate_fn = ConstantLearningRate(learning_rate=learning_rate_config)
            elif isinstance(learning_rate_config, str):
                from networks.learning_rate.learning_rate import get_learning_rate

                self.learning_rate_fn = get_learning_rate(learning_rate_config, learning_rate=0.01)
            else:
                # If it's already a learning rate object
                self.learning_rate_fn = learning_rate_config
            self.epochs_completed = data["epochs_completed"]
            self.data_iteration = data.get("data_iteration", 0)

    def forward(self, x: np.ndarray) -> np.ndarray:
        activations = [x.copy()]
        a = x.copy()
        for i in range(len(self.W)):
            z = self.W[i].T @ a
            if i < len(self.W) - 1:
                a = self.activation.activate(z)
            else:
                a = z
            # a = self.activation.activate(z)
            activations.append(a.copy())
        return activations

    def backward(self, activations: np.ndarray, target: np.ndarray, apply_delta_W: bool = True):
        grads = [np.zeros_like(w) for w in self.W]
        delta = self.loss.calc_delta(target, self.activation.deactivate(activations[-1]))

        for i in reversed(range(len(self.W))):
            grads[i] = activations[i + 1] @ delta

            if i > 0:
                delta = delta @ self.W[i].T * self.activation.deactivate(activations[i])

        if apply_delta_W:
            for i in range(len(self.W)):
                self.delta_W[i] += grads[i]
        else:
            current_rate = self.learning_rate_fn.rate_at_iteration(self.data_iteration)
            for i in range(len(self.W)):
                self.W[i] -= current_rate * grads[i]

    def apply_delta_W(self):
        current_rate = self.learning_rate_fn.rate_at_iteration(self.data_iteration)
        for i in range(len(self.W)):
            self.W[i] -= current_rate * self.delta_W[i]
            self.delta_W[i].fill(0)

    def train(self, data, epochs=1):
        for _ in range(epochs):
            for x_input, y_target in data:
                activations = self.forward(x_input)
                self.backward(activations, y_target, apply_delta_W=False)
                self.data_iteration += 1
            self.apply_delta_W()
            self.epochs_completed += 1

    def test(self, x):
        activations = self.forward(x)
        return activations[-1]

    def show_stats(self):
        logger.info("MLP Network Stats:")

        layer_sizes = [self.input_dim] + self.hidden_dims + [self.output_dim]
        table_print(
            ["layer_sizes", "lr", "epochs completed", "loss_method", "activation_method"],
            [
                [
                    layer_sizes,
                    self.learning_rate_fn.display_name,
                    self.epochs_completed,
                    self.loss.display_name,
                    self.activation.display_name,
                ]
            ],
        )

    def graph_weights(
        self,
        activation_only=True,
        detail="",
        output_dir: Union[pathlib.Path, None] = None,
    ):
        parent_dir = pathlib.Path(output_dir) if output_dir else pathlib.Path("figures")
        parent_dir.mkdir(parents=True, exist_ok=True)
        title = "Activation Structure" if activation_only else "Weight Matrix"
        hslug = "-".join(str(h) for h in self.hidden_dims)
        suffix = f"_{detail}" if detail else ""
        filename = f"mlpnet_in{self.input_dim}_h{hslug}_out{self.output_dim}_" f"{title.replace(' ', '_')}{suffix}.png"
        full_path = parent_dir / filename

        subtitle_parts = [
            f"in={self.input_dim}, hidden=[{hslug}], out={self.output_dim}",
            f"lr={self.learning_rate_fn.display_name}",
        ]
        if detail:
            subtitle_parts.append(detail)
        if activation_only:
            subtitle_parts.append("non-zero weights (dense layers are usually all 1)")

        panel_titles = [f"Layer {i} ({W.shape[0]}×{W.shape[1]})" for i, W in enumerate(self.W)]
        saved_path, fig = render_weight_panels(
            self.W,
            activation_only=activation_only,
            title=title,
            subtitle=" · ".join(subtitle_parts),
            panel_titles=panel_titles,
            path=full_path,
            show=True,
        )
        plt.close(fig)
        return saved_path or str(full_path), fig

    def graph_structure(self, detail="", output_dir: pathlib.Path = None, medium="matplotlib"):
        if medium == "matplotlib":
            self._graph_structure_matplotlib(detail, output_dir)
        else:
            raise ValueError(f"Invalid medium: {medium}")

    def _graph_structure_matplotlib(
        self,
        detail="",
        output_dir: pathlib.Path = None,
        figsize=(6, 6),
        node_radius=0.028,
    ):
        num_vertical_layers = [self.input_dim] + self.hidden_dims + [self.output_dim]
        num_layers = len(num_vertical_layers)
        smallest_vertical_spacing = max(num_vertical_layers) / (figsize[1] * 1.75)

        # map node positions on [0,1]x[0,1]
        node_positions = [
            [
                {
                    "x": i / (num_layers - 1),
                    "y": j / (num_vertical_layers[i] - 1) * smallest_vertical_spacing,
                    "layer": i,
                }
                for j in range(num_vertical_layers[i])
            ]
            for i in range(num_layers)
        ]

        logger.debug(json.dumps(node_positions, indent=4))

        fig, ax = plt.subplots(figsize=figsize)

        # edges
        for u in range(len(node_positions) - 1):
            for v in range(len(node_positions[u])):
                src_node = node_positions[u][v]
                for w in range(len(node_positions[u + 1])):
                    dst_node = node_positions[u + 1][w]
                    ax.plot(
                        [src_node["x"], dst_node["x"]],
                        [src_node["y"], dst_node["y"]],
                        linewidth=1.0,
                        color="black",
                        zorder=1,
                    )

        # nodes
        for u in range(len(node_positions)):
            for v in range(len(node_positions[u])):
                node = node_positions[u][v]
                circ = plt.Circle(
                    (node["x"], node["y"]),
                    node_radius,
                    facecolor="white",
                    edgecolor="black",
                    linewidth=1.2,
                    zorder=10,
                )
                ax.add_patch(circ)

        # tidy margins
        pad = 1.5 * node_radius + 0.2
        ax.axis("off")
        ax.set_aspect("equal")
        ax.set_xlim(0 - pad, 1 + pad)
        ax.set_ylim(0 - pad, 1 + pad)
        plt.tight_layout()
        parent_dir = output_dir if output_dir else pathlib.Path("figures")
        filename = f"mlp_structure{'_' + detail if detail else ''}.png"
        plt.suptitle(f"Graph Structure")
        plt.title(f"lr={self.learning_rate_fn.display_name}, {detail}")
        plt.savefig(parent_dir / filename)
        plt.show()

        return filename

    def train_animated(
        self,
        data: BaseDataset,
        epochs=25,
        pause=0.05,
        output_dir: Union[pathlib.Path, None] = None,
        simple_figure_names: bool = False,
        show_training_metrics: bool = True,
        show_weights_live: bool = False,
    ) -> Tuple[float, float, float]:
        """
        Train while animating loss and regression score over epochs.

        - data: iterable of (x_input, y_target) with shapes (n,) and (n,)
        - simple_figure_names: use short stable names (training_metrics.png,
          weights_live.png) suited for run folders; default keeps descriptive
          names for standalone use.
        - show_training_metrics: live-update the metrics figure each epoch.
        - show_weights_live: open and live-update a weight heatmap each epoch.
        """
        logger.info("Training with params...")
        table_print(["epochs", "num data points"], [[epochs, len(data)]])
        self.figure_service.prepare_training_animation(
            self.training_figure,
            output_dir=output_dir,
            simple_names=simple_figure_names,
            network_kind="MLP",
            display_name=self.display_name,
            loss=self.loss,
            activation=self.activation,
        )
        logger.debug(f"training_figure.filename set to {self.training_figure.filename}")

        weights_figure = None
        if show_weights_live:
            wf_name = "weights_live.png" if simple_figure_names else f"mlpnet_weights_live.png"
            weights_figure = self.figure_service.init_weights_live_figure(
                wf_name,
                f"MLP Weights — {self.display_name}",
                DynamicWeightsFigure.layer_shapes_from_matrices(self.W),
            )
        if not show_training_metrics and show_weights_live:
            # Keep metrics figure data updated for final save, but do not show it live.
            plt.close(self.training_figure.fig)

        # Last-epoch save must not use self.epochs_completed in the condition: we increment it every
        # iteration (unlike Hex), so "epoch == epochs + self.epochs_completed - 1" is wrong for
        # epochs > 1 (e.g. epoch 99 vs 100 + 99 - 1). Align with a fixed half-open range.
        epoch_start = self.epochs_completed
        epoch_stop = epoch_start + epochs

        # training loop
        for epoch in trange(epoch_start, epoch_stop):
            total_loss = 0.0
            self.training_metrics.reset_epoch_tally()

            for x_input, y_target in data:
                activations = self.forward(x_input)
                y_pred = activations[-1]
                total_loss += self.loss.calc_loss(y_target, y_pred)
                self.training_metrics.tally_regression_score_r2(y_pred, y_target)
                self.backward(activations, y_target, apply_delta_W=False)

            epoch_loss = total_loss / len(data)
            epoch_reg_score, epoch_r2, epoch_adj_r2 = self.training_metrics.calc_regression_score_and_r2(
                self._training_metrics_r2_n(), p=self._training_metrics_r2_p()
            )
            self.training_metrics.add_metric(epoch_loss, epoch_reg_score, epoch_r2, epoch_adj_r2)
            self._finalize_training_epoch(
                epoch_loss=epoch_loss,
                epoch_reg_score=epoch_reg_score,
                epoch_r2=epoch_r2,
                epoch_adj_r2=epoch_adj_r2,
                training_channel=0,
                weights_figure=weights_figure,
                pause=pause,
                show_training_metrics=show_training_metrics,
                show_weights_live=show_weights_live,
                is_last_epoch=(epoch == epoch_stop - 1),
            )
            self.epochs_completed += 1

        return epoch_loss, epoch_reg_score, epoch_r2, self.training_figure.fig

    def get_metrics_json(self):
        return self.training_metrics.as_dict()
