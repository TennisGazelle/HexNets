"""Gaussian noise on synthetic datasets (BaseDataset + build_registered_dataset)."""

import numpy as np

from data.dataset import build_registered_dataset
from data.identity_dataset import IdentityDataset


def test_identity_noise_off_matches_clean_baseline() -> None:
    # LinearScale uses global np.random for X; reset between builds for parity.
    np.random.seed(0)
    clean = IdentityDataset(d=3, num_samples=40, noise_seed=99)
    np.random.seed(0)
    noisy_off = IdentityDataset(d=3, num_samples=40, noise_seed=99, noise_mode=None)
    np.testing.assert_array_equal(clean.data["X"], noisy_off.data["X"])
    np.testing.assert_array_equal(clean.data["Y"], noisy_off.data["Y"])


def test_identity_inputs_noise_changes_x_not_y() -> None:
    np.random.seed(123)
    base = IdentityDataset(d=3, num_samples=40, noise_seed=42)
    np.random.seed(123)
    ds = IdentityDataset(
        d=3,
        num_samples=40,
        noise_mode="inputs",
        noise_mu=0.0,
        noise_sigma=0.05,
        noise_seed=42,
    )
    X, Y = ds.data["X"], ds.data["Y"]
    assert not np.allclose(X, base.data["X"])
    np.testing.assert_array_equal(Y, base.data["Y"])


def test_identity_targets_noise_changes_y_not_x() -> None:
    np.random.seed(7)
    base = IdentityDataset(d=3, num_samples=40, noise_seed=7)
    np.random.seed(7)
    ds = IdentityDataset(
        d=3,
        num_samples=40,
        noise_mode="targets",
        noise_mu=0.0,
        noise_sigma=0.05,
        noise_seed=7,
    )
    np.testing.assert_array_equal(ds.data["X"], base.data["X"])
    assert not np.allclose(ds.data["Y"], base.data["Y"])


def test_identity_both_noise_deterministic() -> None:
    def make() -> IdentityDataset:
        np.random.seed(999)
        return IdentityDataset(
            d=2,
            num_samples=25,
            noise_mode="both",
            noise_mu=0.1,
            noise_sigma=0.02,
            noise_seed=123,
        )

    a = make()
    b = make()
    np.testing.assert_array_equal(a.data["X"], b.data["X"])
    np.testing.assert_array_equal(a.data["Y"], b.data["Y"])


def test_build_registered_dataset_noise_kwarg() -> None:
    ds = build_registered_dataset(
        "identity",
        d=4,
        num_samples=12,
        scale=1.0,
        noise_mode="inputs",
        noise_mu=0.0,
        noise_sigma=0.01,
        noise_seed=5,
    )
    assert ds.data["X"].shape == (12, 4)
    assert ds.data["Y"].shape == (12, 4)


def test_build_registered_targets_noise_shape() -> None:
    ds = build_registered_dataset(
        "identity",
        d=3,
        num_samples=15,
        scale=1.0,
        noise_mode="targets",
        noise_mu=0.0,
        noise_sigma=0.03,
        noise_seed=99,
    )
    assert len(ds) == 15
