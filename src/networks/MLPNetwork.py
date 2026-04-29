import logging
import pickle
from typing import List, Union, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pathlib
import json
import copy
from tabulate import tabulate
from tqdm import trange

from services.figure_service.FigureService import FigureService
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
        self._init_figure_service()

        if len(hidden_dims) == 0:
            raise ValueError("hidden_dims must be a list of at least one integer")

        # add layers
        self._add_layer(input_dim, hidden_dims[0])
        for i in range(1, len(hidden_dims)):
            self._add_layer(hidden_dims[i - 1], hidden_dims[i])
        self._add_layer(hidden_dims[-1], output_dim)

    # --- static methods ---
    @staticmethod
    def get_parameter_count(input_dim: int, output_dim: int, hidden_dims: List[int]) -> int:
        if len(hidden_dims) == 0:
            raise ValueError("hidden_dims must be a list of at least one integer")
        dims = [input_dim] + hidden_dims + [output_dim]
        return sum(dims[i] * dims[i + 1] for i in range(len(dims) - 1))

    # --- instance methods ---
    def _init_figure_service(self):
        self.figure_service = FigureService()
        self.figure_service.set_figures_path(None)
        self.training_figure = self.figure_service.init_training_figure(
            f"mlpnet_training_{self.loss}_{self.activation}.png",
            f"Training {self.display_name}",
            self.loss.display_name,
            "mean exp(-RMSE) per example",
            "coefficient of determination",
        )

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
        data = [
            ["layer_sizes", [self.input_dim] + self.hidden_dims + [self.output_dim]],
            ["lr", self.learning_rate_fn.display_name],
            ["epochs completed", self.epochs_completed],
            ["loss_method", self.loss.display_name],
            ["activation_method", self.activation.display_name],
        ]
        logger.info("\n" + tabulate(data, headers=["Parameter", "Value"], tablefmt="grid"))

        # print(f"loss:\t{self.training_metrics['loss'][-1]:.3f}")
        # print(f"regression_score:\t{self.training_metrics.regression_score[-1]:.3f}")
        # print(f"r_squared:\t{self.training_metrics['r_squared'][-1]:.3f}")

    def show_latest_metrics(self):
        metrics = self.training_metrics
        data = (
            [0.0, 0.0, 0.0, 0.0, 0]
            if len(metrics.loss) == 0
            else [
                metrics.loss[-1],
                metrics.regression_score[-1],
                metrics.r_squared[-1],
                metrics.adjusted_r_squared[-1] if metrics.adjusted_r_squared else 0.0,
                self.epochs_completed,
            ]
        )
        table_print(["Loss", "Reg. score", "R^2", "Adjusted R^2", "Epochs"], [[*data]])

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
        filename = (
            f"mlpnet_in{self.input_dim}_h{hslug}_out{self.output_dim}_"
            f"{title.replace(' ', '_')}{suffix}.png"
        )
        full_path = parent_dir / filename

        n_layers = len(self.W)
        fig_w = max(4 * n_layers, 6)
        fig, axes = plt.subplots(1, n_layers, figsize=(fig_w, 5))
        if n_layers == 1:
            axes = np.array([axes])

        vmin = vmax = None
        if not activation_only and self.W:
            vmin = min(float(w.min()) for w in self.W)
            vmax = max(float(w.max()) for w in self.W)

        for ax, i in zip(axes, range(n_layers)):
            W = self.W[i]
            matrix = (W != 0).astype(int) if activation_only else W
            cmap = "Greys" if activation_only else "viridis"
            imshow_kw = {"cmap": cmap, "interpolation": "none"}
            if not activation_only:
                imshow_kw["vmin"] = vmin
                imshow_kw["vmax"] = vmax
            im = ax.imshow(matrix, **imshow_kw)
            ax.set_title(f"Layer {i} ({W.shape[0]}×{W.shape[1]})")
            ax.set_xlabel("out")
            ax.set_ylabel("in")
            if not activation_only:
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        subtitle_parts = [
            f"in={self.input_dim}, hidden=[{hslug}], out={self.output_dim}",
            f"lr={self.learning_rate_fn.display_name}",
        ]
        if detail:
            subtitle_parts.append(detail)
        if activation_only:
            subtitle_parts.append("non-zero weights (dense layers are usually all 1)")
        plt.suptitle(title)
        plt.figtext(0.5, 0.02, " · ".join(subtitle_parts), ha="center", fontsize=9)
        plt.tight_layout(rect=[0, 0.06, 1, 0.96])

        try:
            plt.savefig(full_path)
            # Avoid UserWarning on non-interactive backends (e.g. Agg in CI / tests).
            if "agg" not in str(plt.matplotlib.get_backend()).lower():
                plt.show()
        finally:
            plt.close(fig)

        return str(full_path), fig

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
    ) -> Tuple[float, float, float]:
        """
        Train while animating loss and regression score over epochs.
        - data: iterable of (x_input, y_target) with shapes (n,) and (n,)
        """
        logger.info("MLP Network Training:")
        logger.info(f"epochs:\t{epochs}")
        logger.info(f"datapoints:\t{len(data)}")

        logger.info("Training...")

        logger.debug(f"train_animated called with output_dir={output_dir}")
        self.figure_service.set_figures_path(output_dir)
        logger.debug(f"figures_path set to {self.figure_service.figures_path}")
        # Update the filename to use the new figures_path
        filename_path = (
            pathlib.Path(self.training_figure.filename)
            if isinstance(self.training_figure.filename, str)
            else self.training_figure.filename
        )
        filename_base = filename_path.name
        self.training_figure.filename = self.figure_service.figures_path / filename_base
        logger.debug(f"training_figure.filename set to {self.training_figure.filename}")
        self.training_figure.title = (
            f"MLP Network Training {self.display_name} (loss={self.loss}, activation={self.activation.display_name})"
        )
        self.training_figure.loss_detail = self.loss.display_name
        self.training_figure.regression_score_detail = "mean exp(-RMSE) per example"
        self.training_figure.r2_detail = "coefficient of determination"

        # Last-epoch save must not use self.epochs_completed in the condition: we increment it every
        # iteration (unlike Hex), so "epoch == epochs + self.epochs_completed - 1" is wrong for
        # epochs > 1 (e.g. epoch 99 vs 100 + 99 - 1). Align with a fixed half-open range.
        epoch_start = self.epochs_completed
        epoch_stop = epoch_start + epochs

        # training loop
        for epoch in trange(epoch_start, epoch_stop):
            total_loss = 0.0
            regression_score_sum = 0
            count = 0
            ss_res_sum = 0.0
            sum_y = 0.0
            sum_y2 = 0.0
            num_elems = 0

            for x_input, y_target in data:
                # print('---')
                activations = self.forward(x_input)

                # for li, a in enumerate(activations):
                #     pos = np.count_nonzero(a > 0)
                #     neg = np.count_nonzero(a < 0)
                #     zeros = np.count_nonzero(a == 0)
                #     total = len(a)
                #     print(f"layer {li}: {a}")
                #     print(f"layer {li}: pos={pos}, neg={neg}, zeros={zeros}, total={total}, ratio={pos/total}")

                # loss (MSE on final layer nodes only)
                y_pred = activations[-1]
                total_loss += self.loss.calc_loss(y_target, y_pred)

                # accuracy
                # if classification:
                #     pred_bin = (y_pred >= threshold).astype(int)
                #     tgt_bin = (y_target >= 0.5).astype(int)
                #     correct += np.mean(pred_bin == tgt_bin)
                # else:
                # a simple bounded score for regression feel: 1/(1+MAE)
                # mae = np.mean(np.abs(y_pred - y_target))
                # correct += 1.0 / (1.0 + mae)

                rmse = np.sqrt(np.mean((y_pred - y_target) ** 2))
                score = np.exp(-rmse)  # maps 0->1, larger errors decay smoothly
                regression_score_sum += score

                # scale = np.maximum(1e-8, np.mean(np.abs(y_target)))  # per-sample
                # nmae = np.mean(np.abs(y_pred - y_target)) / scale
                # correct += np.exp(-nmae)

                # r_squared
                ss_res_sum += float(np.sum((y_pred - y_target) ** 2))
                sum_y += float(np.sum(y_target))
                sum_y2 += float(np.sum(y_target**2))

                count += 1
                num_elems += y_pred.shape[0]

                # backprop step
                self.backward(activations, y_target, apply_delta_W=False)

                # if epoch % 10 == 0:
                #     activations_after = self.forward(x0)
                #     y_pred_after = activations_after[-1][self.layer_indices[-1]]
                #     mae_after = np.mean(np.abs(y_pred_after - y_target))
                #     print(f"MAE before: {mae}, after: {mae_after}, delta_y={np.linalg.norm(y_pred_after - y_pred)}")

            # r_squared, cont'd
            ss_tot = sum_y2 - (sum_y**2 / (num_elems))
            r_squared = 1 - (ss_res_sum / (ss_tot + 1e-12))

            # Calculate adjusted R-squared
            p = self.input_dim  # Number of features/parameters
            if num_elems > p + 1:
                adjusted_r_squared = 1 - (1 - r_squared) * (num_elems - 1) / (num_elems - p - 1)
            else:
                adjusted_r_squared = r_squared  # Fallback if insufficient samples

            epoch_loss = total_loss / count
            epoch_reg_score = regression_score_sum / count
            epoch_r2 = r_squared
            epoch_adj_r2 = adjusted_r_squared
            self.training_metrics.add_metric(epoch_loss, epoch_reg_score, epoch_r2, epoch_adj_r2)
            self.training_figure.update_figure(
                {
                    "loss": epoch_loss,
                    "regression_score": epoch_reg_score,
                    "r_squared": epoch_r2,
                    "adjusted_r_squared": epoch_adj_r2,
                }
            )

            self.apply_delta_W()

            plt.pause(pause)

            if epoch == epoch_stop - 1:
                logger.debug(f"About to save figure at epoch {epoch}")
                self.training_figure.save_figure()
                logger.info("Training complete!")
                self.show_latest_metrics()
                logger.info(f"Training figure saved to: {self.training_figure.filename}")

            self.epochs_completed += 1

        return epoch_loss, epoch_reg_score, epoch_r2, self.training_figure.fig

    def get_metrics_json(self):
        return self.training_metrics.as_dict()
