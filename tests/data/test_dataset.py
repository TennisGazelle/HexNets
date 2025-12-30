# a unit test for the linear dataset using pytest


from src.data.dataset import LinearScaleDataset, randomized_enumerate

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
