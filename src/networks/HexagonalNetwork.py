import copy
import math
import os
import pickle
from tabulate import tabulate
from typing import Any, List, Dict, Mapping, Union, Tuple
import pathlib
from utils import table_print
from services.logging_config import get_logger

import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap

from networks.metrics import Metrics
from services.figure_service import FigureService
from services.figure_service.DynamicWeightsFigure import DynamicWeightsFigure
from networks.activation.activations import BaseActivation
from networks.loss.loss import BaseLoss
from networks.network import BaseNeuralNetwork

from networks.activation.LeakyRelu import LeakyReLU
from networks.activation.Relu import ReLU
from networks.activation.Sigmoid import Sigmoid

from networks.loss.HuberLoss import HuberLoss
from networks.loss.LogCoshLoss import LogCoshLoss
from networks.loss.MeanSquaredErrorLoss import MeanSquaredErrorLoss
from networks.loss.QuantileLoss import QuantileLoss
from data.dataset import BaseDataset, randomized_enumerate

# === Hexagonal Neural Network ===
logger = get_logger(__name__)


class HexagonalNeuralNetwork(BaseNeuralNetwork, display_name="hex"):
    def __init__(
        self,
        n: int = 2,
        r: int = 0,
        learning_rate: float = 0.01,
        activation: BaseActivation = Sigmoid,
        loss: BaseLoss = MeanSquaredErrorLoss,
    ):
        super().__init__(learning_rate, activation, loss)
        self.n = n
        self.r = r
        self.total_nodes = self._calc_total_nodes(n)
        self.global_W = self._init_global_W()
        self.dir_W = self._init_dir_W()
        self._sync_global_to_dir()
        self._setup_training_metrics()
        self._init_figure_service()

    # --- static methods ---
    @staticmethod
    def get_run_metadata_schema() -> Mapping[str, Any]:
        """Describe ``model_metadata`` keys persisted in run ``config.json`` for ``model_type`` hex."""
        return {
            "model_type": "hex",
            "keys": {
                "n": {"required": True, "type": "int", "min": 1},
                "r": {"required": True, "type": "int", "min": 0, "max": 5},
                "epr": {
                    "required": False,
                    "type": "int",
                    "description": "Epochs per rotation; when set must satisfy 1 < epr < total epochs.",
                },
                "ro": {
                    "required": False,
                    "type": "list[int]",
                    "description": "Rotation ordering (0..5 each); ignored unless epr is set.",
                },
            },
        }

    @staticmethod
    def validate_epr_ro_fields(
        *,
        epr: int | None,
        ro: list[int] | None,
        epochs: int,
        errors_prefix: str = "Run config: hex model_metadata",
    ) -> None:
        """
        Shared rules for optional ``epr`` / ``ro`` (CLI or ``model_metadata``).

        When ``epr`` is unset and ``ro`` is not ``None``, logs a warning and skips ``ro`` validation.
        When ``epr`` is set, enforces bounds and validates ``ro`` when not ``None``.
        """
        if epr is None:
            if ro is not None:
                logger.warning(
                    "%s: 'ro' (rotation ordering) is ignored because 'epr' (epochs per rotation) is not set.",
                    errors_prefix,
                )
            return

        try:
            epr_i = int(epr)
        except (TypeError, ValueError) as e:
            raise ValueError(f"{errors_prefix}: 'epr' must be an integer.") from e

        if not (1 < epr_i < int(epochs)):
            raise ValueError(
                f"{errors_prefix}: 'epr' must satisfy 1 < epr < epochs " f"(got epr={epr_i}, epochs={int(epochs)})."
            )

        if ro is None:
            return
        if not isinstance(ro, list):
            raise ValueError(f"{errors_prefix}: 'ro' must be a JSON array of integers.")
        for i, x in enumerate(ro):
            try:
                xi = int(x)
            except (TypeError, ValueError) as e:
                raise ValueError(f"{errors_prefix}: 'ro' entry at index {i} must be an integer.") from e
            if not 0 <= xi <= 5:
                raise ValueError(f"{errors_prefix}: 'ro' entries must be in 0..5 (got {xi!r} at index {i}).")

    @staticmethod
    def validate_run_metadata(meta: dict, *, epochs: int) -> None:
        """Validate ``model_metadata`` from a run or train template ``config.json``."""
        if not isinstance(meta, dict):
            raise ValueError("Run config: hex model_metadata must be an object.")
        if "n" not in meta or "r" not in meta:
            raise ValueError("Run config: hex model_metadata requires 'n' and 'r'.")
        try:
            n = int(meta["n"])
            r = int(meta["r"])
        except (TypeError, ValueError) as e:
            raise ValueError("Run config: hex model_metadata 'n' and 'r' must be integers.") from e
        if n < 1:
            raise ValueError("Run config: hex model_metadata 'n' must be >= 1.")
        if not 0 <= r <= 5:
            raise ValueError(f"Run config: hex model_metadata 'r' must be in 0..5 (got {r}).")

        prefix = "Run config: hex model_metadata"
        epr: int | None
        if "epr" in meta and meta["epr"] is not None:
            try:
                epr = int(meta["epr"])
            except (TypeError, ValueError) as e:
                raise ValueError(f"{prefix}: 'epr' must be an integer or null.") from e
        else:
            epr = None

        if epr is None:
            if "ro" in meta and meta["ro"] is not None:
                logger.warning(
                    "%s: 'ro' (rotation ordering) is ignored because 'epr' (epochs per rotation) is not set.",
                    prefix,
                )
            return

        ro: list[int] | None
        if "ro" in meta and meta["ro"] is not None:
            ro_val = meta["ro"]
            if not isinstance(ro_val, list):
                raise ValueError(f"{prefix}: 'ro' must be a JSON array or null.")
            try:
                ro = [int(x) for x in ro_val]
            except (TypeError, ValueError) as e:
                raise ValueError(f"{prefix}: 'ro' must be a JSON array of integers.") from e
        else:
            ro = None

        HexagonalNeuralNetwork.validate_epr_ro_fields(epr=epr, ro=ro, epochs=epochs, errors_prefix=prefix)

    @staticmethod
    def _hex_layer_sizes(n):
        return list(range(n, 2 * n)) + list(range(2 * n - 2, n - 1, -1))

    @staticmethod
    def _get_default_layer_indices(n):
        sizes = HexagonalNeuralNetwork._hex_layer_sizes(n)
        indices = []
        start = 0
        for size in sizes:
            indices.append(list(range(start, start + size)))
            start += size
        return indices

    @staticmethod
    def _get_layer_indices(n, r=0):
        assert 0 <= r <= 5, f"Invalid rotation: {r}"
        max_row_size = 2 * n - 1

        def _get_top_down():
            return HexagonalNeuralNetwork._get_default_layer_indices(n)

        if r == 0:
            return _get_top_down()
        if r == 3:
            return [rw[::-1] for rw in _get_top_down()[::-1]]

        def _get_top_right_to_bottom_left():
            new_indices = [[] for _ in range(max_row_size)]
            node_idx = 0
            for row_idx in range(max_row_size):
                min_new_row_idx = max(0, (row_idx - n + 1))
                max_new_row_idx = min(max_row_size - 1, row_idx + n - 1)
                for new_row_idx in range(min_new_row_idx, max_new_row_idx + 1):
                    new_indices[new_row_idx].append(node_idx)
                    node_idx += 1
            return new_indices

        if r == 4:
            return [rw[::-1] for rw in _get_top_right_to_bottom_left()]
        if r == 1:
            return [rw for rw in _get_top_right_to_bottom_left()[::-1]]

        def _get_top_left_to_bottom_right():
            new_indices = [[] for _ in range(max_row_size)]
            node_idx = 0
            for row_idx in range(max_row_size):
                min_new_row_idx = max(0, (row_idx - n + 1))
                max_new_row_idx = min(max_row_size - 1, row_idx + n - 1)
                for new_row_idx in range(max_new_row_idx, min_new_row_idx - 1, -1):
                    new_indices[new_row_idx].append(node_idx)
                    node_idx += 1
            return new_indices

        if r == 5:
            return [rw for rw in _get_top_left_to_bottom_right()]
        if r == 2:
            return [rw[::-1] for rw in _get_top_left_to_bottom_right()[::-1]]

        raise ValueError(f"Invalid rotation: {r}")

    @staticmethod
    def get_parameter_count(n: int) -> int:
        total = sum(HexagonalNeuralNetwork._hex_layer_sizes(n))
        return total * (total + 1) // 2

    # --- structure helpers ---
    def _calc_total_nodes(self, n):
        return sum(l for l in self._hex_layer_sizes(n))

    def _setup_training_metrics(self):
        self.training_metrics = {i: Metrics() for i in range(0, 6)}

    def get_training_metrics(self, channel: int):
        return self.training_metrics[channel]

    def get_all_training_metrics(self):
        return {i: self.training_metrics[i].as_dict() for i in range(0, 6)}

    def set_training_metrics(self, channel: int, metrics: Metrics):
        self.training_metrics[channel] = metrics

    def _init_figure_service(self):
        from services.figure_service.FigureService import REGRESSION_SCORE_DETAIL, R2_DETAIL

        self.figure_service = FigureService()
        self.figure_service.set_figures_path(None)
        self.training_figure = self.figure_service.init_training_figure(
            f"hexnet_training_{self.loss}_{self.activation}.png",
            f"Training {self.display_name}",
            self.loss.display_name,
            REGRESSION_SCORE_DETAIL,
            R2_DETAIL,
        )

    def _rotation_forward_edge_mask(self) -> np.ndarray:
        """Directed feedforward edges for rotation ``r`` on ``global_W`` indices (row=u, col=v).

        Outlines only (u, v) with u in layer i and v in layer i+1 in ``dir_W[r]['indices']``;
        the transpose cell (v, u) is not marked even though ``global_W`` is symmetric.
        """
        mask = np.zeros(self.global_W.shape, dtype=bool)
        layers = self.dir_W[self.r]["indices"]
        for i in range(len(layers) - 1):
            for u in layers[i]:
                for v in layers[i + 1]:
                    mask[u, v] = True
        return mask

    def get_weight_matrices_for_live_plot(self):
        """Return shared ``global_W`` for live heatmap (full matrix); use highlight mask for active rotation."""
        return [self.global_W.copy()]

    # --- weights ---

    def _init_global_W(self):
        w = np.random.random((self.total_nodes, self.total_nodes)) - 0.5
        w += w.T  # make symmetric
        return w

    def _init_dir_W(self) -> Dict[int, Dict[str, np.ndarray]]:
        dir_metrics = {}
        for i in range(0, 6):
            dir_metrics[i] = {
                "W": np.zeros_like(self.global_W),
                "delta_W": np.zeros_like(self.global_W),
                "indices": self._get_layer_indices(self.n, r=i),
            }
        return dir_metrics

    def _sync_global_to_dir(self):
        for r in range(0, 6):
            r_layer_matrices = self.dir_W[r]["indices"]
            for j in range(len(r_layer_matrices) - 1):
                for u in r_layer_matrices[j]:
                    for v in r_layer_matrices[j + 1]:
                        self.dir_W[r]["W"][u, v] = self.global_W[u, v]

    # --- forward & backward ---
    def pad_input(self, x):
        x0 = np.zeros(self.total_nodes)
        x0[self.dir_W[self.r]["indices"][0]] = x
        return x0

    def pad_output(self, y):
        y0 = np.zeros(self.total_nodes)
        y0[self.dir_W[self.r]["indices"][-1]] = y
        return y0

    def unpad_output(self, y):
        return y[self.dir_W[self.r]["indices"][-1]]

    def forward(self, x: np.ndarray) -> np.ndarray:
        activations = [x.copy()]
        a = x.copy()
        for i in range(len(self.dir_W[self.r]["indices"]) - 1):
            z = self.dir_W[self.r]["W"].T @ a
            if i < len(self.dir_W[self.r]["indices"]) - 2:
                a = self.activation.activate(z)
            else:
                a = z
            activations.append(a.copy())
        return activations

    def backward(self, activations: np.ndarray, target: np.ndarray, apply_delta_W: bool = True):
        grads = np.zeros_like(self.dir_W[self.r]["W"])
        delta = self.loss.calc_delta(target, activations[-1])
        # walk layers backward
        for i in reversed(range(len(self.dir_W[self.r]["indices"]) - 1)):
            src_nodes = self.dir_W[self.r]["indices"][i]
            dst_nodes = self.dir_W[self.r]["indices"][i + 1]
            # weight grads: outer product between delta(dst) and activations(src)
            for u in src_nodes:
                au = activations[i][u]
                # if au == 0:
                #     continue
                for v in dst_nodes:
                    grads[u, v] += delta[v] * au

            # backpropagate delta to previous layer
            # (pre-activation grads already folded into activation deriv)
            if i > 0:
                new_delta = np.zeros(len(activations[i]))
                for u in src_nodes:
                    s = 0.0
                    for v in dst_nodes:
                        s += self.dir_W[self.r]["W"][u, v] * delta[v]
                    new_delta[u] = s * self.activation.deactivate(activations[i][u])
                delta = new_delta
        # SGD update
        # logger.debug(f"||delta_out||={np.linalg.norm(delta):.3e}  ||grads||={np.linalg.norm(grads):.3e}")

        # if apply_delta_W:
        self.dir_W[self.r]["delta_W"] += grads
        # else:
        #     self.dir_metrics[self.r]["W"] -= self.learning_rate * grads

    def apply_delta_W(self):
        current_rate = self.learning_rate_fn.rate_at_iteration(self.data_iteration)
        change = current_rate * self.dir_W[self.r]["delta_W"]
        self.dir_W[self.r]["W"] -= change
        self.global_W -= change

        # clear direction W
        self.dir_W[self.r]["delta_W"].fill(0)

    # --- public API ---
    def per_example_regression_score(self, y_pred, y_target):
        rmse = np.sqrt(np.mean((y_pred - y_target) ** 2))
        score = np.exp(-rmse)  # map rmse to [0,1]
        return score

    def calc_r2(self, y_pred, y_target):
        ss_res = np.sum((y_target - y_pred) ** 2)
        ss_tot = np.sum((y_target - np.mean(y_target)) ** 2)
        return 1 - ss_res / ss_tot

    def rotate(self, direction):
        assert 0 <= direction <= 5, f"Invalid rotation: {direction}"
        self.r = direction % 6
        self._sync_global_to_dir()

    def train(self, data, epochs=1):
        for epoch in range(epochs):
            for x_input, y_target in data:
                x_full = self.pad_input(x_input)
                y_full = self.pad_output(y_target)
                activations = self.forward(x_full)
                self.backward(activations, y_full, apply_delta_W=False)
                self.data_iteration += 1
            self.apply_delta_W()
            self.epochs_completed += 1

    def test(self, x_input):
        x_full = self.pad_input(x_input)
        activations = self.forward(x_full)
        return self.unpad_output(activations[-1])

    def save(self, filepath):
        with open(filepath, "wb") as f:
            pickle.dump(
                {
                    "n": self.n,
                    "r": self.r,
                    "global_W": self.global_W,
                    "dir_metrics": self.dir_W,
                    "training_metrics": self.get_all_training_metrics(),
                    "epochs_completed": self.epochs_completed,
                    "data_iteration": self.data_iteration,
                },
                f,
            )

    def load(self, filepath):
        with open(filepath, "rb") as f:
            state = pickle.load(f)
            self.n = state["n"]
            self.r = state["r"]
            self.global_W = state["global_W"]
            self.total_nodes = self._calc_total_nodes(self.n)
            self.dir_W = state["dir_metrics"]
            self.training_metrics = {i: Metrics(state["training_metrics"][i]) for i in range(0, 6)}
            self.epochs_completed = state["epochs_completed"]
            self.data_iteration = state.get("data_iteration", 0)

    def graph_weights(
        self,
        activation_only=True,
        detail="",
        output_dir: Union[pathlib.Path, None] = None,
    ):
        parent_dir = pathlib.Path(output_dir) if output_dir else pathlib.Path("figures")
        parent_dir.mkdir(parents=True, exist_ok=True)
        title = "Activation Structure" if activation_only else "Weight Matrix"

        filename = f"hexnet_n{self.n}_r{self.r}_{title.replace(' ', '_')}{'_' + detail if detail else ''}.png"
        full_path = parent_dir / filename

        matrix = (self.dir_W[self.r]["W"] != 0).astype(int) if activation_only else self.dir_W[self.r]["W"]

        fig = plt.figure(figsize=(7, 7))
        try:
            plt.imshow(
                matrix,
                cmap="Greys" if activation_only else "viridis",
                interpolation="none",
            )
            plt.suptitle(title)
            plt.title(f"n={self.n}, r={self.r}, lr={self.learning_rate_fn.display_name}, {detail}")
            plt.xticks(np.arange(self.total_nodes))
            plt.yticks(np.arange(self.total_nodes))
            # plt.grid(visible=True, color='black', linewidth=0.5)
            if not activation_only:
                plt.colorbar()
            plt.savefig(full_path)
            plt.show()
        finally:
            plt.close(fig)

        return str(full_path), fig

    def _graph_multi_activation(
        self,
        detail="",
        r_list=list(range(0, 6)),
        output_dir: Union[pathlib.Path, None] = None,
    ):
        title = "Activation Structure"
        filename = f"hexnet_n{self.n}_multi_activation{'_' + detail if detail else ''}.png"

        colors = ["Blues", "Greens", "Reds", "Purples", "Oranges", "Greys"]

        fig = plt.figure(figsize=(3.5 * (self.n - 1), 3.5 * (self.n - 1)))

        try:
            legend_handles = []
            for i, r in enumerate(r_list):
                matrix = (self.dir_W[r]["W"] != 0).astype(int)

                # Create colors with alpha: white (transparent) for 0, colored for 1
                base_cmap = plt.cm.get_cmap(colors[i])  # get the colormap
                colors_with_alpha = [(1, 1, 1, 0)]  # transparent white for 0
                colors_with_alpha.extend([(*base_cmap(0.7)[:3], 0.7)])  # colored with alpha for 1
                custom_cmap = ListedColormap(colors_with_alpha)

                plt.imshow(matrix, cmap=custom_cmap, interpolation="none")
                legend_handles.append(Patch(color=base_cmap(0.7), alpha=0.7, label=f"Rotation {r}"))

            plt.suptitle(title)
            plt.title(f"n={self.n}, {detail}")
            plt.xticks(np.arange(self.total_nodes))
            plt.yticks(np.arange(self.total_nodes))
            plt.legend(handles=legend_handles, title="Rotation")

            parent_dir = pathlib.Path(output_dir) if output_dir else pathlib.Path("figures")
            full_path = parent_dir / filename
            plt.savefig(full_path)

            plt.show()
        finally:
            plt.close(fig)

        return str(full_path), fig

    def _print_indices(self, r):
        for i, layer in enumerate(self.dir_W[r]["indices"]):
            logger.debug(f"layer{i}: {layer}")

    def graph_structure(self, detail="", output_dir=None, medium="matplotlib") -> Tuple[str, plt.Figure]:
        if medium == "matplotlib":
            return self._graph_hex(output_dir, detail=detail)
        elif medium == "dot":
            return self._graph_hex_dot(output_dir)
        else:
            raise ValueError(f"Invalid medium: {medium}")

    def _graph_hex(
        self,
        output_dir: Union[pathlib.Path, None] = None,
        detail="",
        figsize=(10, 10),
        node_radius=0.28,
        edge_alpha=1,
        edge_weighted=True,  # if True, scale edge alpha by |W[u,v]|
        node_fc="white",
        node_ec="black",
        font_size=10,
        label="index",  # "index" (default) or "value"
        values=None,  # optional array aligned to global node index
        dy=math.sqrt(3) / 2,  # vertical spacing
        dx=1.0,  # horizontal spacing
    ):
        """
        Draw a hex-layer network:
        - nodes are arranged in rows with sizes given by self.layer_indices
        - every node in layer i connects to every node in layer i+1
        """
        layers = self.dir_W[self.r]["indices"]  # assumes you've already computed this
        if not layers:
            raise ValueError("layer_indices is empty. Initialize the network first.")

        # --- positions (centered rows, optional stagger for hex look) ---
        node_index = 0
        pos = {}  # node_id -> (x, y)
        for i, layer in enumerate(layers):
            size = len(layer)
            y = -i * dy
            # center the row; optionally stagger alternate rows by 0.5*dx
            offset = -0.5 * (size - 1) * dx
            for j, node in enumerate(layer):
                x = offset + j * dx
                pos[node_index] = (x, y)
                node_index += 1

        # # --- edge weight normalization (if using W) ---
        # def edge_alpha_for(u, v):
        #     if not edge_weighted or getattr(self, "W", None) is None:
        #         return edge_alpha
        #     w = abs(self.W[u, v])
        #     # robust normalization: divide by (median(abs(W_next_layer)) + tiny epsilon)
        #     # so outliers don't blow up visualization
        #     return min(1.0, edge_alpha * (w / (edge_norms.get(i, 1e-9))))

        # edge_norms = {}
        # if edge_weighted and getattr(self, "W", None) is not None:
        #     for i in range(len(layers) - 1):
        #         u_nodes = layers[i]
        #         v_nodes = layers[i + 1]
        #         block = np.abs(self.W[np.ix_(u_nodes, v_nodes)])
        #         med = np.median(block) if block.size else 1.0
        #         edge_norms[i] = med if med > 0 else (np.max(block) if np.max(block) > 0 else 1.0)

        # --- draw ---
        fig, ax = plt.subplots(figsize=figsize)

        # edges
        for i in range(len(layers) - 1):
            for u in layers[i]:
                x1, y1 = pos[u]
                for v in layers[i + 1]:
                    x2, y2 = pos[v]
                    ax.plot(
                        [x1, x2],
                        [y1, y2],
                        linewidth=1.0,
                        color="black",
                        zorder=1,
                        # alpha=edge_alpha_for(u, v),
                    )

        # nodes
        for node, (x, y) in pos.items():
            circ = plt.Circle(
                (x, y),
                node_radius,
                facecolor=node_fc,
                edgecolor=node_ec,
                linewidth=1.2,
                zorder=10,
            )
            ax.add_patch(circ)
            if label == "value" and values is not None:
                txt = (
                    f"{values[node]:.3g}" if isinstance(values[node], (int, float, np.floating)) else str(values[node])
                )
            else:
                txt = str(node)
            ax.text(x, y, txt, ha="center", va="center", fontsize=font_size, zorder=11)

        ax.set_aspect("equal")
        ax.axis("off")

        # tidy margins
        xs, ys = zip(*pos.values())
        pad = 1.5 * node_radius + 0.2
        ax.set_xlim(min(xs) - pad, max(xs) + pad)
        ax.set_ylim(min(ys) - pad, max(ys) + pad)

        plt.tight_layout()
        parent_dir = pathlib.Path(output_dir) if output_dir else pathlib.Path("figures")
        filename = f"hexnet_n{self.n}_r{self.r}_structure{'_' + detail if detail else ''}.png"
        full_path = parent_dir / filename
        plt.suptitle(f"Graph Structure")
        plt.title(f"n={self.n}, r={self.r}, lr={self.learning_rate_fn.display_name}, {detail}")
        try:
            if output_dir:
                plt.savefig(full_path)

            plt.show()
        finally:
            plt.close(fig)

        return str(full_path), fig

    def to_dot_string(self) -> List[str]:
        """
        Export the layered, fully-connected structure as Graphviz DOT text.
        Node IDs are your global indices. Layers are grouped with ranks.
        """
        lines = []
        lines.append("digraph HexNet {")
        lines.append('  graph [rankdir=TB, splines=false, nodesep="0.35", ranksep="0.35"];')
        lines.append("  node  [shape=circle, fontsize=10, width=0.45, fixedsize=true, zorder=10];")
        lines.append('  edge  [penwidth=0.5, dir="none", zorder=1];')

        # group nodes by layer (same rank)
        for i, layer in enumerate(self.dir_W[self.r]["indices"]):
            lines.append(f"  {{ rank=same; // layer {i}")
            for n in layer:
                lines.append(f"    {n};")
            lines.append("  }")

        # invisible edges between nodes in same rank to enforce order
        for i in range(len(self.dir_W[self.r]["indices"])):
            line = f"  {self.dir_W[self.r]['indices'][i][0]}"
            for u_idx, u in enumerate(self.dir_W[self.r]["indices"][i]):
                if u_idx > 0:
                    line += f" -> {u}"
            line += ' [style="invis"];'
            lines.append(line)

        # edges between consecutive layers
        for i in range(len(self.dir_W[self.r]["indices"]) - 1):
            for u in self.dir_W[self.r]["indices"][i]:
                for v in self.dir_W[self.r]["indices"][i + 1]:
                    if getattr(self, "W", None) is not None and False:
                        w = float(self.dir_W[self.r]["W"][u, v])
                        # include a lightweight weight attribute (optional)
                        lines.append(f'  {u} -> {v} [penwidth=1.0, weight="{w:.3g}"];')
                    else:
                        lines.append(f"  {u} -> {v};")

        lines.append("}")
        return lines

    def _graph_hex_dot(self, output_dir: Union[pathlib.Path, None] = None):
        dot_string_list = self.to_dot_string()
        parent_dir = output_dir if output_dir else pathlib.Path("reference")
        dot_file = f"hexnet_n{self.n}_r{self.r}_viewdot.dot"

        with open(parent_dir / dot_file, "w") as f:
            for line in dot_string_list:
                f.write(line + "\n")

        png_file = dot_file.replace(".dot", ".png")
        full_path = parent_dir / png_file
        os.system(f"dot -Tpng {parent_dir / dot_file} -o {full_path}")

        return str(full_path), None

    def show_stats(self):
        logger.info("Hexagonal Network Stats:")
        table_print(
            ["n", "r", "lr", "epochs completed", "loss method", "activation method"],
            [
                [
                    self.n,
                    self.r,
                    self.learning_rate_fn.display_name,
                    self.epochs_completed,
                    self.loss.display_name,
                    self.activation.display_name,
                ]
            ],
        )

    def show_latest_metrics(self):
        metrics = self.training_metrics[self.r]
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
        table_print(
            ["Rotation", "Loss", "Reg. score", "R^2", "Adjusted R^2", "Epochs"],
            [[self.r, *data]],
        )

    def train_animated(
        self,
        data: BaseDataset,
        epochs=25,
        pause=0.05,
        output_dir: Union[pathlib.Path, None] = None,
        simple_figure_names: bool = False,
        show_training_metrics: bool = True,
        show_weights_live: bool = False,
    ):
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
        table_print(
            ["epochs", "rotation", "num data points"],
            [
                [
                    f"{self.epochs_completed} - {self.epochs_completed + epochs}",
                    self.r,
                    len(data),
                ]
            ],
        )

        self.figure_service.prepare_training_animation(
            self.training_figure,
            output_dir=output_dir,
            simple_names=simple_figure_names,
            network_kind="Hexagonal",
            display_name=self.display_name,
            loss=self.loss,
            activation=self.activation,
        )
        logger.debug(f"training_figure.filename set to {self.training_figure.filename}")

        weights_figure = None
        if show_weights_live:
            wf_name = "weights_live.png" if simple_figure_names else "hexnet_weights_live.png"
            weights_figure = self.figure_service.init_weights_live_figure(
                wf_name,
                f"Hexagonal Weights — {self.display_name} r={self.r}",
                DynamicWeightsFigure.layer_shapes_from_matrices(self.get_weight_matrices_for_live_plot()),
            )
        if not show_training_metrics and show_weights_live:
            # Keep metrics figure data updated for final save, but do not show it live.
            plt.close(self.training_figure.fig)

        # training loop
        for epoch in trange(self.epochs_completed, self.epochs_completed + epochs):
            total_loss = 0.0
            # correct = 0
            # count = 0
            # ss_res_sum = 0.0
            # sum_y = 0.0
            # sum_y2 = 0.0

            for index, (x_input, y_target) in randomized_enumerate(data):
                self.data_iteration = index
                # build padded vector
                x_input_full = self.pad_input(x_input)
                y_target_full = self.pad_output(y_target)
                # print("--------------------------------")

                # print(f"x input={x_input}, x_padded={x0}, y expected={y_target}, y_padded={y_full}")

                activations = self.forward(x_input_full)
                # print(f"activations={activations}")

                # for li, a in enumerate(activations):
                #     pos = np.count_nonzero(a > 0)
                #     neg = np.count_nonzero(a < 0)
                #     total = len(a)
                #     if li < len(self.layer_indices) - 1:
                #         print(f"layer {li}: pos={pos}, neg={neg}, total={total}, ratio={pos/total}")

                # loss (MSE on final layer nodes only)
                y_pred = self.unpad_output(activations[-1])
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

                # rmse = np.sqrt(np.mean((y_pred - y_target) ** 2))
                # score = np.exp(-rmse)  # maps 0->1, larger errors decay smoothly
                # correct += score

                # scale = np.maximum(1e-8, np.mean(np.abs(y_target)))  # per-sample
                # nmae = np.mean(np.abs(y_pred - y_target)) / scale
                # correct += np.exp(-nmae)

                # r_squared
                # ss_res_sum += float(np.sum((y_pred - y_target) ** 2))
                # sum_y += float(np.sum(y_target))
                # sum_y2 += float(np.sum(y_target**2))

                # count += 1

                self.training_metrics[self.r].tally_regression_score_r2(y_pred, y_target)

                # backprop step
                self.backward(activations, y_target_full, apply_delta_W=False)

                # if epoch % 10 == 0:
                #     activations_after = self.forward(x0)
                #     y_pred_after = activations_after[-1][self.layer_indices[-1]]
                #     mae_after = np.mean(np.abs(y_pred_after - y_target))
                #     print(f"MAE before: {mae}, after: {mae_after}, delta_y={np.linalg.norm(y_pred_after - y_pred)}")

            # r_squared, cont'd
            # ss_tot = sum_y2 - (sum_y**2 / (count * self.n))
            # r_squared = 1 - (ss_res_sum / (ss_tot + 1e-12))

            epoch_loss = total_loss / len(data)
            epoch_reg_score, epoch_r2, epoch_adj_r2 = self.training_metrics[self.r].calc_regression_score_and_r2(
                self.n, p=self.n
            )
            self.training_metrics[self.r].add_metric(epoch_loss, epoch_reg_score, epoch_r2, epoch_adj_r2)
            self.training_figure.update_figure(
                {
                    "loss": epoch_loss,
                    "regression_score": epoch_reg_score,
                    "r_squared": epoch_r2,
                    "adjusted_r_squared": epoch_adj_r2,
                },
                self.r,
                redraw=show_training_metrics,
            )

            self.apply_delta_W()

            if show_weights_live and weights_figure is not None:
                weights_figure.update_figure(
                    self.get_weight_matrices_for_live_plot(),
                    highlight_masks=[self._rotation_forward_edge_mask()],
                    highlight_channel=self.r,
                )

            if show_training_metrics or show_weights_live:
                plt.pause(pause)

            if epoch == self.epochs_completed + epochs - 1:
                logger.debug(f"About to save figure at epoch {epoch}")
                self.training_figure.save_figure()
                logger.info("")
                logger.info("Training complete!")
                self.show_latest_metrics()
                logger.info(f"Training figure saved to: {self.training_figure.filename}")
                if weights_figure is not None:
                    weights_figure.save_figure()
                    logger.info(f"Weights figure saved to: {weights_figure.filename}")

        # todo: keep track of epochs completed for each rotation
        if self.r == 0:
            self.epochs_completed += epochs

        return epoch_loss, epoch_reg_score, epoch_r2, self.training_figure.fig

    def get_metrics_json(self):
        return self.get_all_training_metrics()
