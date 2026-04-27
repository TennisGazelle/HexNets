"""Unit tests for networks.metrics.Metrics regression score and R² accumulation."""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np
import pytest

from networks.metrics import Metrics

Pair = Tuple[np.ndarray, np.ndarray]


def _reference_r_squared_adjusted(
    pairs: Sequence[Pair],
    n: int,
    p: int,
    eps: float = 1e-12,
) -> Tuple[float, float]:
    """Mirror tally + calc R² / adjusted R² from (y_pred, y_target) pairs (same as metrics.py)."""
    y_sum = 0.0
    y2_sum = 0.0
    ss_res_sum = 0.0
    count = 0
    for y_pred, y_target in pairs:
        y_pred = np.asarray(y_pred, dtype=float)
        y_target = np.asarray(y_target, dtype=float)
        assert y_pred.shape == y_target.shape == (n,)
        diff_squared = (y_pred - y_target) ** 2
        ss_res_sum += float(np.sum(diff_squared))
        y_sum += float(np.sum(y_target))
        y2_sum += float(np.sum(y_target**2))
        count += 1
    ss_tot = y2_sum - (y_sum**2 / (count * n))
    r_squared = 1.0 - (ss_res_sum / (ss_tot + eps))
    num_samples = count * n
    if num_samples > p + 1:
        adjusted_r_squared = 1.0 - (1.0 - r_squared) * (num_samples - 1) / (num_samples - p - 1)
    else:
        adjusted_r_squared = r_squared
    return r_squared, adjusted_r_squared


def _finalize_metrics(pairs: Sequence[Pair], n: int, p: int | None = None) -> Tuple[float, float, float]:
    m = Metrics()
    for y_pred, y_target in pairs:
        m.tally_regression_score_r2(y_pred, y_target)
    return m.calc_regression_score_and_r2(n, p)


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

    ref_r2, ref_adj = _reference_r_squared_adjusted([(p1, y1), (p2, y2)], n=2, p=2)
    assert r2 == pytest.approx(ref_r2)
    assert adj == pytest.approx(ref_adj)

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


def test_metrics_r2_adjusted_matches_reference_on_toy_pairs():
    """Sanity: Metrics finalization matches duplicated reference for arbitrary small data."""
    pairs: List[Pair] = [
        (np.array([0.0, 2.0]), np.array([1.0, 1.0])),
        (np.array([1.0, 0.0]), np.array([1.0, 3.0])),
    ]
    _, r2, adj = _finalize_metrics(pairs, n=2, p=1)
    ref_r2, ref_adj = _reference_r_squared_adjusted(pairs, n=2, p=1)
    assert r2 == pytest.approx(ref_r2)
    assert adj == pytest.approx(ref_adj)


def test_perfect_fit_multiple_samples_r_squared_and_adjusted_one():
    """Several tallies with y_pred == y_target and N > p + 1 → R² = 1, adjusted R² = 1."""
    pairs: List[Pair] = []
    for _ in range(5):
        yt = np.array([1.5, -0.25])
        pairs.append((yt.copy(), yt.copy()))
    _, r2, adj = _finalize_metrics(pairs, n=2, p=1)
    assert r2 == pytest.approx(1.0)
    assert adj == pytest.approx(1.0)
    ref_r2, ref_adj = _reference_r_squared_adjusted(pairs, n=2, p=1)
    assert r2 == pytest.approx(ref_r2)
    assert adj == pytest.approx(ref_adj)


def test_mean_baseline_poor_fit_r_squared_near_zero():
    """Predict every target scalar as the grand mean → pooled R² = 0 when N > p + 1."""
    pairs: List[Pair] = [
        (np.array([2.0]), np.array([1.0])),
        (np.array([2.0]), np.array([2.0])),
        (np.array([2.0]), np.array([3.0])),
    ]
    _, r2, adj = _finalize_metrics(pairs, n=1, p=1)
    assert r2 == pytest.approx(0.0, abs=1e-9)
    ref_r2, ref_adj = _reference_r_squared_adjusted(pairs, n=1, p=1)
    assert r2 == pytest.approx(ref_r2)
    assert adj == pytest.approx(ref_adj)
    # N=3, p=1 → adjusted uses formula and differs from r² (here r²=0 → adj = 1 - 2/1 = -1)
    assert adj == pytest.approx(-1.0)


def test_worse_than_mean_negative_r_squared():
    """Errors larger than predicting the grand mean → negative R² (hand-checked R² = -2)."""
    pairs: List[Pair] = [
        (np.array([5.0]), np.array([1.0])),
        (np.array([1.0]), np.array([3.0])),
        (np.array([3.0]), np.array([5.0])),
    ]
    _, r2, adj = _finalize_metrics(pairs, n=1, p=1)
    ref_r2, ref_adj = _reference_r_squared_adjusted(pairs, n=1, p=1)
    assert r2 == pytest.approx(ref_r2)
    assert r2 < 0
    assert adj == pytest.approx(ref_adj)
    assert r2 == pytest.approx(-2.0)


def test_low_sample_num_elems_le_p_plus_one_adjusted_equals_r_squared():
    """When count * n <= p + 1, adjusted R² falls back to R²."""
    pairs: List[Pair] = [(np.array([1.0, 2.0]), np.array([0.0, 1.0]))]
    _, r2, adj = _finalize_metrics(pairs, n=2, p=5)
    assert 2 <= 5 + 1
    assert adj == pytest.approx(r2)
    _, ref_adj = _reference_r_squared_adjusted(pairs, n=2, p=5)
    assert adj == pytest.approx(ref_adj)


def test_boundary_num_samples_equals_p_plus_two_adjusted_uses_formula():
    """Smallest N with N > p + 1 is N = p + 2; adjusted R² must use the standard correction."""
    p = 2
    pairs: List[Pair] = [
        (np.array([1.0, 0.0]), np.array([1.0, 1.0])),
        (np.array([0.0, 1.0]), np.array([1.0, 0.0])),
    ]
    n = 2
    assert len(pairs) * n == p + 2
    _, r2, adj = _finalize_metrics(pairs, n=n, p=p)
    ref_r2, ref_adj = _reference_r_squared_adjusted(pairs, n=n, p=p)
    assert r2 == pytest.approx(ref_r2)
    assert adj == pytest.approx(ref_adj)
    assert r2 == pytest.approx(-3.0)
    assert adj == pytest.approx(-11.0)


def test_constant_targets_perfect_predictions_r_squared_one():
    """Degenerate SS_tot (constant y): perfect preds → R² = 1."""
    pairs: List[Pair] = [
        (np.array([7.0]), np.array([7.0])),
        (np.array([7.0]), np.array([7.0])),
    ]
    _, r2, adj = _finalize_metrics(pairs, n=1, p=1)
    assert r2 == pytest.approx(1.0)
    assert adj == pytest.approx(1.0)


def test_constant_targets_wrong_predictions_r_squared_via_eps():
    """Degenerate SS_tot: SS_res / (ss_tot + eps) with ss_tot = 0 → huge negative R²."""
    pairs: List[Pair] = [
        (np.array([0.0]), np.array([1.0])),
        (np.array([0.0]), np.array([1.0])),
        (np.array([0.0]), np.array([1.0])),
    ]
    eps = 1e-12
    ss_res = 3.0
    expected_r2 = 1.0 - ss_res / eps
    _, r2, _ = _finalize_metrics(pairs, n=1, p=1)
    assert r2 == pytest.approx(expected_r2)
    ref_r2, _ = _reference_r_squared_adjusted(pairs, n=1, p=1, eps=eps)
    assert r2 == pytest.approx(ref_r2)
