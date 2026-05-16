"""Family A dirname parsing and hex/mlp pair index (Streamlit-free)."""

from pathlib import Path

import pytest

from hexnets_web.pages.run_browser.fam_a_signatures import (
    FamASignature,
    build_fam_a_pair_index,
    dropdown_options_for_field,
    parse_fam_a_dirname,
    resolve_picker_signature,
)


def test_parse_fam_a_dirname_hex_ok() -> None:
    out = parse_fam_a_dirname("hex-affine-leaky_relu-huber-constant-n3")
    assert out is not None
    model, sig = out
    assert model == "hex"
    assert sig == FamASignature("affine", "leaky_relu", "huber", "constant", "n3")


def test_parse_fam_a_dirname_mlp_ok() -> None:
    out = parse_fam_a_dirname("mlp-full_linear-sigmoid-mean_squared_error-constant-n4")
    assert out is not None
    model, sig = out
    assert model == "mlp"
    assert sig == FamASignature("full_linear", "sigmoid", "mean_squared_error", "constant", "n4")


@pytest.mark.parametrize(
    "name",
    [
        "hex-short",
        "hex-a-b-c-d",
        "other-affine-leaky_relu-huber-constant-n3",
        "hex-affine-leaky_relu-huber-constant-x3",
        "hex-affine-leaky_relu-huber-constant-n",
        "orphan",
    ],
)
def test_parse_fam_a_dirname_rejects(name: str) -> None:
    assert parse_fam_a_dirname(name) is None


def _touch_run(p: Path) -> None:
    p.mkdir(parents=True)
    for n in ("config.json", "manifest.json", "training_metrics.json"):
        (p / n).write_text("{}", encoding="utf-8")


def test_build_fam_a_pair_index_pair_hex_only_mlp_only(tmp_path: Path) -> None:
    root = tmp_path / "e2etest-famA"
    root.mkdir()

    sig = FamASignature("affine", "leaky_relu", "huber", "constant", "n3")
    hex_dir = root / "hex-affine-leaky_relu-huber-constant-n3"
    mlp_dir = root / "mlp-affine-leaky_relu-huber-constant-n3"
    hex_only = root / "hex-identity-sigmoid-mean_squared_error-constant-n4"
    mlp_only = root / "mlp-identity-relu-huber-exponential_decay-n4"

    _touch_run(hex_dir)
    _touch_run(mlp_dir)
    _touch_run(hex_only)
    _touch_run(mlp_only)

    bad = root / "hex-not-e2e-layout"
    bad.mkdir()
    (bad / "config.json").write_text("{}", encoding="utf-8")

    index = build_fam_a_pair_index(root)
    assert len(index) == 3

    pair = index[sig]
    assert pair["hex"] == hex_dir
    assert pair["mlp"] == mlp_dir

    h_only_sig = FamASignature("identity", "sigmoid", "mean_squared_error", "constant", "n4")
    assert index[h_only_sig]["hex"] == hex_only
    assert index[h_only_sig]["mlp"] is None

    m_only_sig = FamASignature("identity", "relu", "huber", "exponential_decay", "n4")
    assert index[m_only_sig]["hex"] is None
    assert index[m_only_sig]["mlp"] == mlp_only


def test_resolve_picker_signature_empty_index() -> None:
    with pytest.raises(ValueError, match="empty"):
        resolve_picker_signature({}, {"dataset": "a"})


def test_resolve_picker_signature_coerces(tmp_path: Path) -> None:
    root = tmp_path / "famA"
    root.mkdir()
    _touch_run(root / "hex-a-b-c-d-n1")
    index = build_fam_a_pair_index(root)
    sig = resolve_picker_signature(
        index,
        {
            "dataset": "wrong",
            "activation": "",
            "loss": "",
            "learning_rate": "",
            "n_token": "",
        },
    )
    assert sig == FamASignature("a", "b", "c", "d", "n1")


def test_dropdown_options_for_field_filters() -> None:
    s1 = FamASignature("a", "relu", "huber", "constant", "n4")
    s2 = FamASignature("a", "sigmoid", "huber", "constant", "n4")
    s3 = FamASignature("b", "relu", "mse", "constant", "n4")
    index = {
        s1: {"hex": None, "mlp": None},
        s2: {"hex": None, "mlp": None},
        s3: {"hex": None, "mlp": None},
    }
    assert dropdown_options_for_field(index, "dataset") == ["a", "b"]
    assert dropdown_options_for_field(index, "activation", dataset="a") == ["relu", "sigmoid"]
    assert dropdown_options_for_field(index, "loss", dataset="a", activation="relu") == ["huber"]
