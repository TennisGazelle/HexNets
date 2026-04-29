"""Run Browser discovers training curve PNGs for both Hex and MLP runs."""

from pathlib import Path

from hexnets_web.run_browser import _training_plot_paths


def test_training_plot_paths_finds_mlp_png(tmp_path: Path) -> None:
    plots = tmp_path / "plots"
    plots.mkdir()
    mlp_png = plots / "mlpnet_training_mean_squared_error_sigmoid.png"
    mlp_png.write_bytes(b"\x89PNG\r\n\x1a\n")

    found = _training_plot_paths(tmp_path)
    assert found == [mlp_png]


def test_training_plot_paths_finds_hex_and_mlp_sorted(tmp_path: Path) -> None:
    plots = tmp_path / "plots"
    plots.mkdir()
    hex_png = plots / "hexnet_training_mean_squared_error_relu.png"
    mlp_png = plots / "mlpnet_training_mean_squared_error_sigmoid.png"
    hex_png.write_bytes(b"\x89PNG\r\n\x1a\n")
    mlp_png.write_bytes(b"\x89PNG\r\n\x1a\n")

    found = _training_plot_paths(tmp_path)
    assert found == sorted([hex_png, mlp_png])


def test_training_plot_paths_empty_when_no_plots_dir(tmp_path: Path) -> None:
    assert _training_plot_paths(tmp_path) == []
