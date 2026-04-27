"""Run manifest ingestion, JSON errors, and legacy run loading."""

from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import pytest

from networks.HexagonalNetwork import HexagonalNeuralNetwork
from networks.activation.activations import get_activation_function
from networks.loss.loss import get_loss_function
from services.run_service.RunService import RunService
from utils import read_json_object

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_read_json_object_invalid_json_includes_path_and_hint(tmp_path: Path) -> None:
    bad = tmp_path / "broken.json"
    bad.write_text("{ not valid json", encoding="utf-8")
    with pytest.raises(ValueError) as exc_info:
        read_json_object(bad, "config.json")
    msg = str(exc_info.value)
    assert "not valid JSON" in msg
    assert str(bad.resolve()) in msg
    assert "line" in msg


def test_normalize_dataset_missing_legacy_keys_raises() -> None:
    cfg = {"model_type": "hex"}
    with pytest.raises(ValueError, match="dataset"):
        RunService._normalize_dataset_in_config(cfg)


def test_normalize_dataset_idempotent_when_nested_present() -> None:
    cfg = {"dataset": {"id": "identity", "num_samples": 10, "scale": None}}
    RunService._normalize_dataset_in_config(cfg)
    assert cfg["dataset"]["id"] == "identity"


def test_get_parameter_count_hex_n2() -> None:
    net = HexagonalNeuralNetwork(
        n=2,
        r=0,
        learning_rate="constant",
        activation=get_activation_function("sigmoid"),
        loss=get_loss_function("mean_squared_error"),
    )
    assert (
        HexagonalNeuralNetwork.get_parameter_count(2)
        == net.total_nodes * (net.total_nodes + 1) // 2
    )


@pytest.mark.skipif(
    not (REPO_ROOT / "runs" / "e2etest-hex-train" / "config.json").exists(),
    reason="fixture run e2etest-hex-train not present",
)
def test_legacy_config_json_normalizes_dataset_block() -> None:
    """Legacy runs lack ``dataset``; normalization matches flat keys (no full pickle load)."""
    cfg_path = REPO_ROOT / "runs" / "e2etest-hex-train" / "config.json"
    cfg = read_json_object(cfg_path, "config.json")
    RunService._normalize_dataset_in_config(cfg)
    assert cfg["dataset"]["id"] == cfg["dataset_type"]
    assert cfg["dataset"]["num_samples"] == cfg["dataset_size"]


def test_load_run_bad_config_json_wraps_run_dir(tmp_path: Path) -> None:
    run_dir = tmp_path / "fake-run"
    run_dir.mkdir()
    (run_dir / "config.json").write_text("{", encoding="utf-8")
    (run_dir / "manifest.json").write_text("{}", encoding="utf-8")
    (run_dir / "training_metrics.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError) as exc_info:
        RunService(Namespace(run_dir=run_dir))
    assert "Failed to ingest run" in str(exc_info.value)


def test_load_run_config_missing_required_keys_raises(tmp_path: Path) -> None:
    run_dir = tmp_path / "r"
    run_dir.mkdir()
    (run_dir / "config.json").write_text(
        '{"dataset_type": "identity", "dataset_size": 10}',
        encoding="utf-8",
    )
    (run_dir / "manifest.json").write_text("{}", encoding="utf-8")
    (run_dir / "training_metrics.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError) as exc_info:
        RunService(Namespace(run_dir=run_dir))
    assert "missing required keys" in str(exc_info.value)


@patch(
    "services.run_service.RunService.resolve_git_commit", return_value=("a" * 40, None)
)
def test_new_run_manifest_has_traceability_fields(
    _mock_git: object, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(RunService, "runs_dir", tmp_path)
    args = Namespace(
        run_name="manifest-test",
        run_dir=None,
        model="hex",
        n=2,
        rotation=0,
        learning_rate="constant",
        epochs=1,
        type="identity",
        dataset_size=20,
        seed=99,
        loss="mean_squared_error",
        activation="sigmoid",
        run_note="bench run",
        run_tags="paper,v1",
        dataset_scale=None,
    )
    run = RunService(args)
    m = run.manifest_contents
    assert m["schema_version"] == 1
    assert m["git_commit"] == "a" * 40
    assert m["random_seed"] == 99
    assert m["run_note"] == "bench run"
    assert m["run_tags"] == ["paper", "v1"]
    assert m["trainable_parameter_count"] > 0
    assert run.config_contents["schema_version"] == 1
    assert run.config_contents["random_seed"] == 99
    assert run.config_contents["dataset"]["id"] == "identity"
    assert run.config_contents["dataset"]["scale"] is None
