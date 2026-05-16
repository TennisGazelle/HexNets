"""Parse e2e Family A run directory names and index hex/mlp pairs (no Streamlit)."""

from __future__ import annotations

import pathlib
import re
from collections.abc import Mapping
from typing import NamedTuple

from hexnets_web.pages.run_browser.run_validation import is_valid_run_dir

# Family A e2e layout: ``hex|mlp`` + ``dataset-activation-loss-learning_rate-n{int}``.
# Dataset ids in ``scripts/e2e-bench-A.sh`` use underscores, not hyphens; extra hyphens
# in ``dataset`` would make this split ambiguous without a different encoding or config fallback.

_N_TOKEN = re.compile(r"^n\d+$")


class FamASignature(NamedTuple):
    dataset: str
    activation: str
    loss: str
    learning_rate: str
    n_token: str


def parse_fam_a_dirname(dirname: str) -> tuple[str, FamASignature] | None:
    """
    Parse ``hex-…`` / ``mlp-…`` names matching e2e-bench-A ``combo_string`` layout.

    Returns ``(model, signature)`` with ``model`` in ``{"hex", "mlp"}``, or ``None``.
    """
    if "-" not in dirname:
        return None
    model, rest = dirname.split("-", 1)
    if model not in ("hex", "mlp"):
        return None
    parts = rest.split("-")
    if len(parts) != 5:
        return None
    dataset, activation, loss, learning_rate, n_token = parts
    if not _N_TOKEN.match(n_token):
        return None
    return (model, FamASignature(dataset, activation, loss, learning_rate, n_token))


def build_fam_a_pair_index(root: pathlib.Path) -> dict[FamASignature, dict[str, pathlib.Path | None]]:
    """
    Map each parsed signature to ``{"hex": path|None, "mlp": path|None}`` for valid run dirs.

    Only immediate children of ``root`` that pass :func:`is_valid_run_dir` and parse with
    :func:`parse_fam_a_dirname` are included. Duplicate basenames for the same model+signature
    are not expected; the last directory seen wins.
    """
    out: dict[FamASignature, dict[str, pathlib.Path | None]] = {}
    if not root.is_dir():
        return out
    for child in sorted(root.iterdir(), key=lambda p: p.name.lower()):
        if not child.is_dir() or not is_valid_run_dir(child):
            continue
        parsed = parse_fam_a_dirname(child.name)
        if parsed is None:
            continue
        model, sig = parsed
        slot = out.setdefault(sig, {"hex": None, "mlp": None})
        slot[model] = child
    return out


def resolve_picker_signature(
    index: dict[FamASignature, dict[str, pathlib.Path | None]],
    picked: Mapping[str, str],
) -> FamASignature:
    """
    Return a signature that exists in ``index``, greedily aligning with ``picked`` field-wise.

    ``picked`` keys: ``dataset``, ``activation``, ``loss``, ``learning_rate``, ``n_token``.
    Raises ``ValueError`` if ``index`` is empty.
    """
    keys = sorted(index.keys())
    if not keys:
        raise ValueError("index is empty")
    cand: list[FamASignature] = list(keys)
    field_map = (
        ("dataset", "dataset"),
        ("activation", "activation"),
        ("loss", "loss"),
        ("learning_rate", "learning_rate"),
        ("n_token", "n_token"),
    )
    for attr, pick_key in field_map:
        opts = sorted({getattr(k, attr) for k in cand})
        if not opts:
            return keys[0]
        want = picked.get(pick_key, "")
        choice = want if want in opts else opts[0]
        cand = [k for k in cand if getattr(k, attr) == choice]
        if not cand:
            return keys[0]
    return cand[0]


def dropdown_options_for_field(
    index: dict[FamASignature, dict[str, pathlib.Path | None]],
    field: str,
    *,
    dataset: str | None = None,
    activation: str | None = None,
    loss: str | None = None,
    learning_rate: str | None = None,
) -> list[str]:
    """Sorted distinct values for ``field`` among signatures matching the given prefix filters."""
    cand = list(index.keys())
    if dataset is not None:
        cand = [s for s in cand if s.dataset == dataset]
    if activation is not None:
        cand = [s for s in cand if s.activation == activation]
    if loss is not None:
        cand = [s for s in cand if s.loss == loss]
    if learning_rate is not None:
        cand = [s for s in cand if s.learning_rate == learning_rate]
    return sorted({getattr(s, field) for s in cand})
