"""Shared matplotlib helpers for static weight / activation-structure heatmaps."""

from __future__ import annotations

import pathlib
from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np


def render_weight_panels(
    weight_matrices: List[np.ndarray],
    *,
    activation_only: bool = False,
    title: str,
    subtitle: Optional[str] = None,
    panel_titles: Optional[List[str]] = None,
    path: Optional[Union[pathlib.Path, str]] = None,
    show: bool = True,
    figsize_per_panel: float = 4.0,
    min_fig_width: float = 6.0,
) -> Tuple[Optional[str], plt.Figure]:
    """Render one heatmap per weight matrix and optionally save to *path*."""
    n_layers = len(weight_matrices)
    fig_w = max(figsize_per_panel * n_layers, min_fig_width)
    fig, axes = plt.subplots(1, n_layers, figsize=(fig_w, 5))
    if n_layers == 1:
        axes = np.array([axes])

    vmin = vmax = None
    if not activation_only and weight_matrices:
        vmin = min(float(w.min()) for w in weight_matrices)
        vmax = max(float(w.max()) for w in weight_matrices)

    cmap = "Greys" if activation_only else "viridis"
    for ax, i in zip(axes, range(n_layers)):
        W = weight_matrices[i]
        matrix = (W != 0).astype(int) if activation_only else W
        imshow_kw = {"cmap": cmap, "interpolation": "none", "aspect": "auto"}
        if not activation_only:
            imshow_kw["vmin"] = vmin
            imshow_kw["vmax"] = vmax
        im = ax.imshow(matrix, **imshow_kw)
        if panel_titles is not None and i < len(panel_titles):
            ax.set_title(panel_titles[i])
        elif not activation_only:
            ax.set_title(f"Layer {i} ({W.shape[0]}×{W.shape[1]})")
        else:
            ax.set_title(f"{W.shape[0]}×{W.shape[1]}")
        ax.set_xlabel("out")
        ax.set_ylabel("in")
        if not activation_only:
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.suptitle(title)
    if subtitle:
        plt.figtext(0.5, 0.02, subtitle, ha="center", fontsize=9)
        plt.tight_layout(rect=[0, 0.06, 1, 0.96])
    else:
        plt.tight_layout()

    saved_path: Optional[str] = None
    try:
        if path is not None:
            full_path = pathlib.Path(path)
            full_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(full_path)
            saved_path = str(full_path)
        if show and "agg" not in str(plt.matplotlib.get_backend()).lower():
            plt.show()
    finally:
        plt.close(fig)

    return saved_path, fig
