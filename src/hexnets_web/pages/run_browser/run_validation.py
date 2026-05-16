"""Pure helpers to recognize a HexNets run directory on disk (no Streamlit)."""

from __future__ import annotations

import pathlib

REQUIRED_RUN_ROOT_JSON = (
    "config.json",
    "manifest.json",
    "training_metrics.json",
)


def missing_run_artifacts(path: pathlib.Path) -> list[str]:
    """Return basenames of required JSON files that are missing or not regular files."""
    if not path.is_dir():
        return list(REQUIRED_RUN_ROOT_JSON)
    missing: list[str] = []
    for name in REQUIRED_RUN_ROOT_JSON:
        p = path / name
        if not p.is_file():
            missing.append(name)
    return missing


def is_valid_run_dir(path: pathlib.Path) -> bool:
    """True if ``path`` is a directory with config, manifest, and training_metrics JSON."""
    return len(missing_run_artifacts(path)) == 0


def run_class_from_dir_name(dirname: str) -> str:
    """Bucket for Family A-style names: ``hex-...`` / ``mlp-...`` / ``other``."""
    if "-" in dirname:
        return dirname.split("-", 1)[0]
    return "other"


def discover_valid_runs_under(root: pathlib.Path) -> dict[str, list[pathlib.Path]]:
    """Immediate subdirectories of ``root`` that pass :func:`is_valid_run_dir`, grouped by class."""
    grouped: dict[str, list[pathlib.Path]] = {}
    if not root.is_dir():
        return grouped
    for child in sorted(root.iterdir(), key=lambda p: p.name.lower()):
        if not child.is_dir():
            continue
        if not is_valid_run_dir(child):
            continue
        cls = run_class_from_dir_name(child.name)
        grouped.setdefault(cls, []).append(child)
    return grouped
