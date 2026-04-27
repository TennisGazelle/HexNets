# a unit test for the linear dataset using pytest

import numpy as np

from data.dataset import (
    DATASET_FUNCTIONS,
    LinearScaleDataset,
    build_datasets_glossary_parent,
    build_registered_dataset,
    list_registered_dataset_display_names,
    randomized_enumerate,
)

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
