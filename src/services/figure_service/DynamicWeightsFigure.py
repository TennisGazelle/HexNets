import pathlib
from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from services.figure_service.figure import Figure


class DynamicWeightsFigure(Figure):
    """Live-updating weight heatmap figure.

    Works for any network: pass one shape per weight matrix (MLP yields one
    subplot per layer; Hex yields a single subplot for the global weight
    matrix).  Call update_figure() each epoch to refresh the display, and
    save_figure() at the end to persist the final frame.

    Optional ``highlight_masks`` (one per panel, same shape as the matrix)
    draws unfilled rectangles over True cells (e.g. hex active rotation edges).
    Pass ``highlight_channel`` (e.g. hex rotation ``r``) to draw all outlines in
    the same tab10 color as ``TrainingFigure`` uses for that channel.
    """

    save_log_label = "weights figure"

    def __init__(self, title: str, filename: str, layer_shapes: List[Tuple[int, int]]):
        super().__init__(filename)
        self.title = title
        self.layer_shapes = tuple((int(h), int(w)) for h, w in layer_shapes)

        n_layers = len(layer_shapes)
        fig_w = max(4 * n_layers, 6)
        self.fig, axes = plt.subplots(1, n_layers, figsize=(fig_w, 5))
        if n_layers == 1:
            axes = np.array([axes])
        self.axes = axes
        self.colorbars = []
        self.images: List[plt.Artist] = []
        self._highlight_patches: List[List[Rectangle]] = [[] for _ in range(n_layers)]

        for ax, shape in zip(axes, layer_shapes):
            data = np.zeros(shape)
            im = ax.imshow(data, cmap="viridis", interpolation="none", aspect="auto")
            ax.set_xlabel("out")
            ax.set_ylabel("in")
            cb = self.fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            self.images.append(im)
            self.colorbars.append(cb)

        self.fig.suptitle(title)
        plt.tight_layout()

    # --- static methods ---

    @staticmethod
    def layer_shapes_from_matrices(weight_matrices: List[np.ndarray]) -> List[Tuple[int, int]]:
        """Convenience: extract shapes from a list of weight matrices."""
        return [W.shape for W in weight_matrices]

    # --- instance methods ---

    def _clear_highlight_patches(self, panel_index: int) -> None:
        ax = self.axes[panel_index]
        for p in self._highlight_patches[panel_index]:
            p.remove()
        self._highlight_patches[panel_index].clear()

    def _apply_highlight_mask(
        self,
        panel_index: int,
        mask: np.ndarray,
        matrix: np.ndarray,
        *,
        edgecolor,
    ) -> None:
        ax = self.axes[panel_index]
        if mask.shape != matrix.shape:
            raise ValueError(f"highlight mask shape {mask.shape} does not match matrix shape {matrix.shape}")
        n = max(matrix.shape)
        linewidth = float(max(0.8, min(2.5, 40.0 / max(n, 1))))
        for i, j in np.argwhere(mask):
            rect = Rectangle(
                (j - 0.5, i - 0.5),
                1.0,
                1.0,
                fill=False,
                edgecolor=edgecolor,
                linewidth=linewidth,
                zorder=10,
            )
            ax.add_patch(rect)
            self._highlight_patches[panel_index].append(rect)

    def update_figure(
        self,
        weight_matrices: List[np.ndarray],
        *,
        activation_only: bool = False,
        highlight_masks: Optional[List[Optional[np.ndarray]]] = None,
        highlight_channel: Optional[int] = None,
    ):
        if highlight_masks is None:
            highlight_masks = [None] * len(weight_matrices)
        if len(highlight_masks) != len(weight_matrices):
            raise ValueError("highlight_masks must align with weight_matrices")

        if highlight_channel is not None:
            hi = int(highlight_channel) % len(self.colors)
            highlight_edgecolor = self.colors[hi]
        else:
            highlight_edgecolor = None

        if activation_only:
            matrices = [(W != 0).astype(float) for W in weight_matrices]
            vmin, vmax = 0.0, 1.0
            cmap = "Greys"
        else:
            matrices = [W.copy() for W in weight_matrices]
            all_vals = np.concatenate([m.ravel() for m in matrices])
            vmin = float(all_vals.min())
            vmax = float(all_vals.max()) if all_vals.max() != all_vals.min() else vmin + 1e-6
            cmap = "magma"

        for k, (im, ax, matrix) in enumerate(zip(self.images, self.axes, matrices)):
            self._clear_highlight_patches(k)
            im.set_data(matrix)
            im.set_clim(vmin, vmax)
            im.set_cmap(cmap)
            n_rows, n_cols = matrix.shape
            ax.set_title(f"{n_rows}×{n_cols}")
            hm = highlight_masks[k]
            if hm is not None:
                ec = highlight_edgecolor if highlight_edgecolor is not None else "k"
                self._apply_highlight_mask(k, np.asarray(hm, dtype=bool), matrix, edgecolor=ec)

        self._canvas_draw()

    @staticmethod
    def export_matrices_to_path(
        weight_matrices: List[np.ndarray],
        path: Union[pathlib.Path, str],
        *,
        title: str,
        activation_only: bool = False,
        panel_titles: Optional[List[str]] = None,
    ) -> str:
        """Persist weight matrices using the shared static heatmap renderer."""
        from services.figure_service.weight_heatmap import render_weight_panels

        saved_path, _ = render_weight_panels(
            weight_matrices,
            activation_only=activation_only,
            title=title,
            panel_titles=panel_titles,
            path=path,
            show=False,
        )
        return saved_path or str(path)
