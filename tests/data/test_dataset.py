# a unit test for the linear dataset using pytest


from src.data.dataset import LinearDataset, randomized_enumerate

def test_linear_dataset():
    dataset = LinearDataset(d=2, num_samples=100, scale=1.0)
    assert len(dataset) == 100

    for x, y in dataset:
        assert x[0] == y[0]
        assert x[1] == y[1]

def test_randomizing_dataset():
    dataset = LinearDataset(d=2, num_samples=10, scale=1.0)

    seen_indices = set()
    for index, (x, y) in randomized_enumerate(dataset):
        assert index not in seen_indices
        seen_indices.add(index)
    
    assert len(seen_indices) == len(dataset)
    print(seen_indices)
