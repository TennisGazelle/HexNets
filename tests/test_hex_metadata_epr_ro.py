"""Tests for hex ``model_metadata`` ``epr`` / ``ro`` validation and CLI merge."""

from __future__ import annotations

import json
import logging
import sys
from argparse import Namespace
from pathlib import Path

import pytest

from networks.HexagonalNetwork import HexagonalNeuralNetwork
from services.run_config import RunConfig


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


def test_validate_run_metadata_epr_ro_ok() -> None:
    meta = {"n": 3, "r": 0, "epr": 10, "ro": [0, 5, 3]}
    HexagonalNeuralNetwork.validate_run_metadata(meta, epochs=50)


def test_validate_run_metadata_ro_without_epr_warns(caplog: pytest.LogCaptureFixture) -> None:
    meta = {"n": 3, "r": 0, "ro": [0, 1]}
    with caplog.at_level(logging.WARNING):
        HexagonalNeuralNetwork.validate_run_metadata(meta, epochs=50)
    assert "ignored" in caplog.text.lower()


def test_validate_run_metadata_epr_out_of_range() -> None:
    meta = {"n": 3, "r": 0, "epr": 1}
    with pytest.raises(ValueError, match="1 < epr < epochs"):
        HexagonalNeuralNetwork.validate_run_metadata(meta, epochs=50)

    meta["epr"] = 50
    with pytest.raises(ValueError, match="1 < epr < epochs"):
        HexagonalNeuralNetwork.validate_run_metadata(meta, epochs=50)


def test_validate_run_metadata_ro_invalid_when_epr_set() -> None:
    meta = {"n": 3, "r": 0, "epr": 10, "ro": [0, 6]}
    with pytest.raises(ValueError, match="0..5"):
        HexagonalNeuralNetwork.validate_run_metadata(meta, epochs=50)


def test_run_config_ro_without_epr_warns(caplog: pytest.LogCaptureFixture) -> None:
    cfg = _minimal_hex_config()
    cfg["model_metadata"] = {"n": 3, "r": 0, "ro": [1, 2]}
    with caplog.at_level(logging.WARNING):
        RunConfig(cfg).validate()
    assert "ignored" in caplog.text.lower()


def test_run_config_to_namespace_hex_epr_ro() -> None:
    cfg = _minimal_hex_config(
        model_metadata={"n": 3, "r": 1, "epr": 5, "ro": [0, 1]},
    )
    ns = RunConfig(cfg).to_namespace()
    assert ns.epr == 5
    assert ns.ro == [0, 1]


def test_run_config_to_namespace_hex_epr_unset_ro_cleared() -> None:
    cfg = _minimal_hex_config(model_metadata={"n": 3, "r": 1, "ro": [9, 9]})
    ns = RunConfig(cfg).to_namespace()
    assert ns.epr is None
    assert ns.ro is None


def test_merge_cli_overrides_epr_ro(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(
        json.dumps(_minimal_hex_config(epochs=100, model_metadata={"n": 3, "r": 0, "epr": 20})),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["hexnet", "train", "--run-config", str(cfg_path), "--epr", "11", "--ro", "0", "5", "3"],
    )

    original = Namespace(
        run_config=cfg_path,
        run_config_json=None,
        run_dir=None,
        run_name=None,
        run_note=None,
        run_tags=None,
    )
    merged = RunConfig.from_cli_sources(original).merged_train_namespace(original=original)
    assert merged.epochs == 100
    assert merged.epr == 11
    assert merged.ro == [0, 5, 3]
