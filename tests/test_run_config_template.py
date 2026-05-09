"""Tests for ``--run-config`` / ``--run-config-json`` template loading and CLI overrides."""

from __future__ import annotations

import json
import sys
from argparse import Namespace
from pathlib import Path

import pytest

from commands.run_config_template import (
    merge_run_config_template_args,
    parse_explicit_cli_overrides,
    run_config_to_namespace,
    validate_run_config_cli_exclusivity,
)


def _minimal_hex_config(**patch: object) -> dict:
    cfg = {
        "schema_version": 1,
        "model_type": "hex",
        "activation_type": "sigmoid",
        "loss_type": "mean_squared_error",
        "learning_rate": "constant",
        "epochs": 50,
        "dataset_type": "identity",
        "dataset_size": 100,
        "dataset": {
            "id": "identity",
            "num_samples": 100,
            "scale": None,
            "noise": None,
        },
        "random_seed": 7,
        "run_folder_name": "legacy-run",
        "model_metadata": {"n": 3, "r": 2},
    }
    cfg.update(patch)
    return cfg


def test_run_config_to_namespace_hex() -> None:
    ns = run_config_to_namespace(_minimal_hex_config())
    assert ns.model == "hex"
    assert ns.epochs == 50
    assert ns.n == 3
    assert ns.rotation == 2
    assert ns.seed == 7
    assert ns.type == "identity"
    assert ns.dataset_size == 100
    assert ns.dataset_noise is None
    assert ns.run_dir is None


def test_run_config_learning_rate_numeric_becomes_constant_string() -> None:
    ns = run_config_to_namespace(_minimal_hex_config(learning_rate=0.01))
    assert ns.learning_rate == "constant"


def test_run_config_dataset_noise_round_trip() -> None:
    cfg = _minimal_hex_config()
    cfg["dataset"]["noise"] = {"mode": "inputs", "mu": 0.1, "sigma": 0.2}
    ns = run_config_to_namespace(cfg)
    assert ns.dataset_noise == "inputs"
    assert ns.dataset_noise_mu == pytest.approx(0.1)
    assert ns.dataset_noise_sigma == pytest.approx(0.2)


def test_validate_exclusivity_config_and_json() -> None:
    args = Namespace(run_config=Path("/tmp/x.json"), run_config_json="{}", run_dir=None)
    with pytest.raises(ValueError, match="only one"):
        validate_run_config_cli_exclusivity(args)


def test_validate_exclusivity_config_and_run_dir(tmp_path: Path) -> None:
    cfg_path = tmp_path / "c.json"
    cfg_path.write_text(json.dumps(_minimal_hex_config()), encoding="utf-8")
    args = Namespace(run_config=cfg_path, run_config_json=None, run_dir=tmp_path / "run")
    with pytest.raises(ValueError, match="run-dir"):
        validate_run_config_cli_exclusivity(args)


def test_merge_cli_overrides_epochs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps(_minimal_hex_config(epochs=50)), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["hexnet", "train", "--run-config", str(cfg_path), "-e", "99"])

    original = Namespace(
        run_config=cfg_path,
        run_config_json=None,
        run_dir=None,
        run_name="merged-test",
        run_note=None,
        run_tags=None,
    )
    merged = merge_run_config_template_args(original)
    assert merged.epochs == 99
    assert merged.run_config is None
    assert merged.run_dir is None
    assert merged.run_name == "merged-test"


def test_merge_keeps_file_epochs_without_cli_flag(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps(_minimal_hex_config(epochs=77)), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["hexnet", "train", "--run-config", str(cfg_path)])

    original = Namespace(
        run_config=cfg_path,
        run_config_json=None,
        run_dir=None,
        run_name=None,
        run_note=None,
        run_tags=None,
    )
    merged = merge_run_config_template_args(original)
    assert merged.epochs == 77


def test_parse_explicit_overrides_ignores_unknown_tokens() -> None:
    ns = parse_explicit_cli_overrides(["--run-config", "ignored.json", "-e", "3"])
    assert ns.epochs == 3


def test_merge_from_json_string(monkeypatch: pytest.MonkeyPatch) -> None:
    raw = json.dumps(_minimal_hex_config(epochs=10))
    monkeypatch.setattr(sys, "argv", ["hexnet", "train", "--run-config-json", raw, "-n", "4"])

    original = Namespace(
        run_config=None,
        run_config_json=raw,
        run_dir=None,
        run_name=None,
        run_note=None,
        run_tags=None,
    )
    merged = merge_run_config_template_args(original)
    assert merged.epochs == 10
    assert merged.n == 4


def test_merge_overrides_run_name_from_cli(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps(_minimal_hex_config()), encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        ["hexnet", "train", "--run-config", str(cfg_path), "--run_name", "cli-name"],
    )

    original = Namespace(
        run_config=cfg_path,
        run_config_json=None,
        run_dir=None,
        run_name="from-first-parse",
        run_note=None,
        run_tags=None,
    )
    merged = merge_run_config_template_args(original)
    assert merged.run_name == "cli-name"
