"""Unit tests for networks.metrics.Metrics regression score and R² accumulation."""

import numpy as np
import pytest

from networks.metrics import Metrics


def test_tally_and_finalize_mean_exp_neg_rmse_and_r_squared():
    """Two examples, n=2: regression score = mean(exp(-rmse)); R² matches manual totals."""
    m = Metrics()
    y1 = np.array([1.0, 3.0])
    p1 = np.array([1.0, 3.0])
    y2 = np.array([1.0, 3.0])
    p2 = np.array([1.0, 1.0])

    m.tally_regression_score_r2(p1, y1)
    m.tally_regression_score_r2(p2, y2)

    reg, r2, adj = m.calc_regression_score_and_r2(n=2, p=2)

    s1 = 1.0
    rmse2 = float(np.sqrt(np.mean((p2 - y2) ** 2)))
    s2 = float(np.exp(-rmse2))
    assert reg == pytest.approx((s1 + s2) / 2)

    ss_res = float(np.sum((p1 - y1) ** 2) + np.sum((p2 - y2) ** 2))
    y_flat = np.concatenate([y1, y2])
    ss_tot = float(np.sum(y_flat**2) - (np.sum(y_flat) ** 2) / y_flat.size)
    assert r2 == pytest.approx(1.0 - ss_res / (ss_tot + 1e-12))

    num_elems = m.count * 2
    assert num_elems == 4
    expected_adj = 1.0 - (1.0 - r2) * (num_elems - 1) / (num_elems - 2 - 1)
    assert adj == pytest.approx(expected_adj)


def test_perfect_predictions_r_squared_one():
    m = Metrics()
    y = np.array([0.5, -1.0])
    m.tally_regression_score_r2(y.copy(), y.copy())
    reg, r2, adj = m.calc_regression_score_and_r2(n=2, p=2)
    assert reg == pytest.approx(1.0)
    assert r2 == pytest.approx(1.0)
    # num_elems = 2, p = 2 -> not > p+1, adjusted falls back to r2
    assert adj == pytest.approx(r2)
