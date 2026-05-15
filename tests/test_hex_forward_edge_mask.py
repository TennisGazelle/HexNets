"""Tests for hex live-weights forward-edge mask (global_W + rotation DAG)."""

import numpy as np

from networks.HexagonalNetwork import HexagonalNeuralNetwork
from networks.activation.activations import get_activation_function
from networks.loss.loss import get_loss_function


def _expected_forward_edge_count(net: HexagonalNeuralNetwork) -> int:
    layers = net.dir_W[net.r]["indices"]
    total = 0
    for i in range(len(layers) - 1):
        total += len(layers[i]) * len(layers[i + 1])
    return total


def test_rotation_forward_edge_mask_matches_layer_connectivity() -> None:
    net = HexagonalNeuralNetwork(
        n=3,
        r=1,
        learning_rate="constant",
        activation=get_activation_function("sigmoid"),
        loss=get_loss_function("mean_squared_error"),
    )
    mask = net._rotation_forward_edge_mask()
    assert mask.shape == net.global_W.shape
    assert mask.dtype == bool
    assert int(mask.sum()) == _expected_forward_edge_count(net)


def test_rotation_forward_edge_mask_not_all_true() -> None:
    net = HexagonalNeuralNetwork(
        n=4,
        r=0,
        learning_rate="constant",
        activation=get_activation_function("relu"),
        loss=get_loss_function("mean_squared_error"),
    )
    mask = net._rotation_forward_edge_mask()
    assert not mask.all()
    assert mask.sum() > 0


def test_get_weight_matrices_for_live_plot_uses_global_W() -> None:
    net = HexagonalNeuralNetwork(
        n=2,
        r=0,
        learning_rate="constant",
        activation=get_activation_function("sigmoid"),
        loss=get_loss_function("mean_squared_error"),
    )
    mats = net.get_weight_matrices_for_live_plot()
    assert len(mats) == 1
    assert mats[0].shape == net.global_W.shape
    np.testing.assert_array_equal(mats[0], net.global_W)
    mats[0][0, 0] = 999.0
    assert net.global_W[0, 0] != 999.0
