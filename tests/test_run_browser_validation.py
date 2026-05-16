"""Run directory validation and Family A grouping helpers."""

from pathlib import Path

from hexnets_web.pages.run_browser.run_validation import (
    discover_valid_runs_under,
    is_valid_run_dir,
    missing_run_artifacts,
    run_class_from_dir_name,
)


def test_missing_run_artifacts_empty_dir(tmp_path: Path) -> None:
    assert missing_run_artifacts(tmp_path) == [
        "config.json",
        "manifest.json",
        "training_metrics.json",
    ]


def test_is_valid_run_dir_false_when_any_json_missing(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "manifest.json").write_text("{}", encoding="utf-8")
    assert is_valid_run_dir(tmp_path) is False
    miss = missing_run_artifacts(tmp_path)
    assert miss == ["training_metrics.json"]


def test_is_valid_run_dir_true_when_all_present(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "manifest.json").write_text("{}", encoding="utf-8")
    (tmp_path / "training_metrics.json").write_text("[]", encoding="utf-8")
    assert is_valid_run_dir(tmp_path) is True
    assert missing_run_artifacts(tmp_path) == []


def test_run_class_from_dir_name() -> None:
    assert run_class_from_dir_name("hex-identity-sigmoid-mse") == "hex"
    assert run_class_from_dir_name("mlp-identity-sigmoid-mse") == "mlp"
    assert run_class_from_dir_name("orphan") == "other"


def test_discover_valid_runs_under_groups_and_skips_invalid(tmp_path: Path) -> None:
    root = tmp_path / "e2etest-famA"
    root.mkdir()

    good_hex = root / "hex-a"
    good_hex.mkdir()
    for name in ("config.json", "manifest.json", "training_metrics.json"):
        (good_hex / name).write_text("{}", encoding="utf-8")

    bad = root / "hex-broken"
    bad.mkdir()
    (bad / "config.json").write_text("{}", encoding="utf-8")

    other = root / "customrun"
    other.mkdir()
    for name in ("config.json", "manifest.json", "training_metrics.json"):
        (other / name).write_text("{}", encoding="utf-8")

    grouped = discover_valid_runs_under(root)
    assert set(grouped.keys()) == {"hex", "other"}
    assert len(grouped["hex"]) == 1
    assert grouped["hex"][0].name == "hex-a"
    assert len(grouped["other"]) == 1
    assert grouped["other"][0].name == "customrun"
