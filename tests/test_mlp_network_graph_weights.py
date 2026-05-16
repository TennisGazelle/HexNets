"""MLP weight / activation-structure figures written by graph_weights."""

import matplotlib

matplotlib.use("Agg")

from pathlib import Path

from data.identity_dataset import IdentityDataset
from networks.MLPNetwork import MLPNetwork
from networks.activation.activations import get_activation_function
from networks.loss.loss import get_loss_function


def test_graph_weights_writes_activation_and_weight_pngs(tmp_path: Path) -> None:
    net = MLPNetwork(
        input_dim=2,
        output_dim=2,
        hidden_dims=[3],
        learning_rate="constant",
        activation=get_activation_function("sigmoid"),
        loss=get_loss_function("mean_squared_error"),
    )
    out_dir = tmp_path / "plots"
    out_dir.mkdir()

    path_act, _ = net.graph_weights(activation_only=True, output_dir=out_dir)
    path_w, _ = net.graph_weights(activation_only=False, detail="trained", output_dir=out_dir)

    p_act = Path(path_act)
    p_w = Path(path_w)
    assert p_act.is_file() and p_act.stat().st_size > 0
    assert p_w.is_file() and p_w.stat().st_size > 0
    assert "Activation_Structure" in p_act.name
    assert "Weight_Matrix" in p_w.name
    assert "trained" in p_w.name


def test_train_animated_saves_training_curve_png_for_multi_epoch_run(tmp_path: Path, monkeypatch) -> None:
    """Regression: save must run on last epoch even when epochs > 1 (epochs_completed advances each epoch)."""
    monkeypatch.setattr("networks.MLPNetwork.plt.pause", lambda *args, **kwargs: None)

    net = MLPNetwork(
        input_dim=3,
        output_dim=3,
        hidden_dims=[4],
        learning_rate="constant",
        activation=get_activation_function("sigmoid"),
        loss=get_loss_function("mean_squared_error"),
    )
    plots = tmp_path / "plots"
    plots.mkdir()
    data = IdentityDataset(d=3, num_samples=10)
    net.train_animated(data, epochs=3, pause=0, output_dir=plots)
    training_pngs = sorted(plots.glob("*net_training_*.png"))
    assert len(training_pngs) == 1
    assert training_pngs[0].stat().st_size > 0


def test_train_animated_simple_names_uses_stable_basename(tmp_path: Path, monkeypatch) -> None:
    """With simple_figure_names=True the output file is training_metrics.png."""
    monkeypatch.setattr("networks.MLPNetwork.plt.pause", lambda *args, **kwargs: None)

    net = MLPNetwork(
        input_dim=3,
        output_dim=3,
        hidden_dims=[4],
        learning_rate="constant",
        activation=get_activation_function("sigmoid"),
        loss=get_loss_function("mean_squared_error"),
    )
    plots = tmp_path / "plots"
    plots.mkdir()
    data = IdentityDataset(d=3, num_samples=5)
    net.train_animated(data, epochs=2, pause=0, output_dir=plots, simple_figure_names=True)
    assert (plots / "training_metrics.png").is_file()
    assert (plots / "training_metrics.png").stat().st_size > 0


def test_train_animated_weights_live_saves_weights_png(tmp_path: Path, monkeypatch) -> None:
    """show_weights_live=True produces weights_live.png alongside training_metrics.png."""
    monkeypatch.setattr("networks.MLPNetwork.plt.pause", lambda *args, **kwargs: None)

    net = MLPNetwork(
        input_dim=2,
        output_dim=2,
        hidden_dims=[3],
        learning_rate="constant",
        activation=get_activation_function("sigmoid"),
        loss=get_loss_function("mean_squared_error"),
    )
    plots = tmp_path / "plots"
    plots.mkdir()
    data = IdentityDataset(d=2, num_samples=5)
    net.train_animated(
        data, epochs=2, pause=0, output_dir=plots,
        simple_figure_names=True, show_weights_live=True,
    )
    assert (plots / "training_metrics.png").is_file()
    assert (plots / "weights_live.png").is_file()
    assert (plots / "weights_live.png").stat().st_size > 0
    # Regression: metrics should still be accumulated even when not shown live.
    assert len(net.training_figure.training_metrics[0]["loss"]) == 2
