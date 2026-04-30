"""Training plot path discovery (no Streamlit; safe for unit tests)."""

from __future__ import annotations

import pathlib


def _training_plot_paths(run_path: pathlib.Path) -> list[pathlib.Path]:
    plots_dir = run_path / "plots"
    if not plots_dir.is_dir():
        return []
    return sorted(plots_dir.glob("*net_training_*.png"))
