# a unit test for the linear dataset using pytest

import numpy as np
import pytest

from data.dataset import (
    DATASET_FUNCTIONS,
    build_datasets_glossary_parent,
    build_registered_dataset,
    list_registered_dataset_display_names,
    randomized_enumerate,
)
from data.linear_scale_dataset import LinearScaleDataset

def test_linear_dataset():
    dataset = LinearScaleDataset(d=2, num_samples=100, scale=1.0)
    assert len(dataset) == 100

    for x, y in dataset:
        assert x[0] == y[0]
        assert x[1] == y[1]

def test_randomizing_dataset():
    dataset = LinearScaleDataset(d=2, num_samples=10, scale=1.0)

    iter_1 = []
    for index, (x, y) in randomized_enumerate(dataset):
        assert index not in iter_1
        iter_1.append(index)

    iter_2 = []
    for index, (x, y) in randomized_enumerate(dataset):
        assert index not in iter_2
        iter_2.append(index)

    assert iter_1 != iter_2
    assert len(iter_1) == len(dataset)
    assert len(iter_2) == len(dataset)


def test_list_registered_dataset_display_names_includes_diagonal_scale():
    names = list_registered_dataset_display_names()
    assert "diagonal_scale" in names
    assert names == sorted(names)


def test_build_registered_dataset_diagonal_scale_mapping():
    ds = build_registered_dataset("diagonal_scale", d=3, num_samples=50, scale=1.0)
    assert len(ds) == 50
    x, y = ds[0]
    factors = np.arange(1, 4, dtype=float)
    np.testing.assert_allclose(y, x * factors)
    assert x.shape == (3,) and y.shape == (3,)


def test_build_datasets_glossary_parent_matches_registry():
    parent = build_datasets_glossary_parent()
    assert parent.title == "Datasets"
    assert len(parent.children) == len(DATASET_FUNCTIONS)
    all_aliases = [a for child in parent.children for a in child.aliases]
    assert "diagonal_scale" in all_aliases
    assert "identity" in all_aliases
    for child in parent.children:
        assert child.title
        assert child.english


@pytest.mark.parametrize("display_name", sorted(DATASET_FUNCTIONS.keys()))
def test_build_registered_dataset_smoke(display_name: str):
    ds = build_registered_dataset(display_name, d=4, num_samples=20, scale=1.0)
    assert len(ds) == 20
    x, y = ds[0]
    assert x.shape == (4,) and y.shape == (4,)


def test_orthogonal_rotation_preserves_norm():
    from data.orthogonal_rotation_dataset import OrthogonalRotationDataset

    ds = OrthogonalRotationDataset(d=5, num_samples=30, scale=1.0, seed=0)
    for x, y in ds:
        np.testing.assert_allclose(np.linalg.norm(x), np.linalg.norm(y), rtol=1e-5, atol=1e-5)


def test_simplex_projection_row_sums_to_one():
    from data.simplex_projection_dataset import SimplexProjectionDataset

    ds = SimplexProjectionDataset(d=4, num_samples=15, scale=1.0, seed=1)
    _, y = ds[0]
    np.testing.assert_allclose(y.sum(), 1.0, rtol=1e-5, atol=1e-5)
    assert (y >= -1e-9).all()


def test_unit_sphere_projection_unit_norm():
    from data.unit_sphere_projection_dataset import UnitSphereProjectionDataset

    ds = UnitSphereProjectionDataset(d=3, num_samples=10, scale=1.0, seed=2)
    for _, y in ds:
        np.testing.assert_allclose(np.linalg.norm(y), 1.0, rtol=1e-5, atol=1e-5)


def test_sparse_identity_sparsity():
    from data.sparse_identity_dataset import SparseIdentityDataset

    ds = SparseIdentityDataset(d=10, num_samples=5, scale=1.0, seed=3)
    for x, y in ds:
        assert np.count_nonzero(x) <= 3
        np.testing.assert_array_equal(x, y)


def test_fixed_permutation_recovers_with_inverse():
    from data.fixed_permutation_dataset import FixedPermutationDataset

    ds = FixedPermutationDataset(d=4, num_samples=8, seed=42)
    inv = np.empty_like(ds._perm)
    inv[ds._perm] = np.arange(ds.d)
    for x, y in ds:
        np.testing.assert_array_equal(x, y[inv])


def test_import_dataset_subprocess_does_not_import_streamlit():
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    src = str(root / "src")
    cmd = (
        f"import sys; sys.path.insert(0, {src!r}); import data.dataset; "
        "assert 'streamlit' not in sys.modules"
    )
    r = subprocess.run([sys.executable, "-c", cmd], cwd=str(root), capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
