# Testing Guide

This document covers testing patterns, conventions, and how to add tests.

## Current Test Structure

```
tests/
  data/
    test_dataset.py    # Dataset tests
```

**Note:** Test coverage is currently minimal. This guide outlines patterns for expanding test coverage.

## Test Organization

### Directory Structure

Tests mirror source structure:

```
src/
  networks/
    activation/
      Sigmoid.py
tests/
  networks/
    activation/
      test_sigmoid.py
```

### Naming Conventions

- Test files: `test_*.py`
- Test classes: `Test*` (optional, for grouping)
- Test functions: `test_*`

## Testing Patterns

### Component Tests

Test individual components (loss functions, activations, etc.):

```python
# tests/networks/loss/test_mean_squared_error.py
import numpy as np
import pytest
from networks.loss.MeanSquaredErrorLoss import MeanSquaredErrorLoss

def test_mse_loss_calculation():
    loss_fn = MeanSquaredErrorLoss()
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.1, 1.9, 3.1])
    
    loss = loss_fn.calc_loss(y_true, y_pred)
    expected = np.mean((y_true - y_pred) ** 2)
    
    assert np.isclose(loss, expected)

def test_mse_delta_calculation():
    loss_fn = MeanSquaredErrorLoss()
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.1, 1.9, 3.1])
    
    delta = loss_fn.calc_delta(y_true, y_pred)
    expected = 2 * (y_pred - y_true) / y_true.shape[0]
    
    np.testing.assert_array_almost_equal(delta, expected)
```

### Network Tests

Test network forward/backward passes:

```python
# tests/networks/test_hexagonal_network.py
import numpy as np
import pytest
from networks.HexagonalNetwork import HexagonalNeuralNetwork
from networks.activation.Sigmoid import Sigmoid
from networks.loss.MeanSquaredErrorLoss import MeanSquaredErrorLoss

def test_forward_pass():
    net = HexagonalNeuralNetwork(
        n=3,
        r=0,
        activation=Sigmoid(),
        loss=MeanSquaredErrorLoss()
    )
    
    x = np.array([0.5, 0.3, 0.7])
    activations = net.forward(x)
    
    assert len(activations) > 0
    assert activations[-1].shape == (3,)

def test_backward_pass():
    net = HexagonalNeuralNetwork(n=3, r=0, ...)
    x = np.array([0.5, 0.3, 0.7])
    y = np.array([0.6, 0.4, 0.8])
    
    activations = net.forward(x)
    net.backward(activations, y)
    
    # Check that weights were updated
    assert net.global_W is not None
```

### CLI Tests

Test CLI argument parsing and validation:

```python
# tests/commands/test_train_command.py
import pytest
from argparse import Namespace
from commands.train_command import TrainCommand

def test_train_command_validation():
    command = TrainCommand()
    args = Namespace(
        n=3,
        rotation=0,
        model="hex",
        activation="sigmoid",
        loss="mean_squared_error",
        learning_rate="constant",
        epochs=10,
        pause=0.05,
        type="identity",
        dataset_size=100,
        seed=42,
        run_name=None,
        run_dir=None
    )
    
    # Should not raise
    command.validate_args(args)

def test_train_command_invalid_n():
    command = TrainCommand()
    args = Namespace(n=1, ...)  # n < 2
    
    with pytest.raises(ValueError, match="at least 2"):
        command.validate_args(args)
```

### Integration Tests

Test end-to-end workflows:

```python
# tests/integration/test_training_workflow.py
import pytest
from run_service import RunService
from argparse import Namespace

def test_training_creates_run():
    args = Namespace(
        n=3, rotation=0, model="hex",
        activation="sigmoid", loss="mean_squared_error",
        learning_rate="constant", epochs=5,
        pause=0, type="identity", dataset_size=50,
        seed=42, run_name="test_run", run_dir=None
    )
    
    run = RunService(args)
    assert run.run_folder_path.exists()
    assert run.config_path.exists()
    assert run.manifest_path.exists()
```

## Test Utilities

### Fixtures

Use pytest fixtures for common setup:

```python
# conftest.py
import pytest
import numpy as np
from networks.HexagonalNetwork import HexagonalNeuralNetwork

@pytest.fixture
def sample_network():
    return HexagonalNeuralNetwork(
        n=3,
        r=0,
        activation=Sigmoid(),
        loss=MeanSquaredErrorLoss()
    )

@pytest.fixture
def sample_data():
    return {
        "X": np.random.rand(10, 3),
        "Y": np.random.rand(10, 3)
    }
```

### Test Data

Create test data generators:

```python
def generate_test_data(n_samples=10, n_dims=3):
    X = np.random.rand(n_samples, n_dims)
    Y = X * 2  # Simple transformation
    return {"X": X, "Y": Y}
```

## What to Test

### High Priority

1. **Component Registration**
   - Components register with correct display names
   - Discovery functions return all components
   - Factory functions create correct instances

2. **Network Forward/Backward**
   - Forward pass produces correct shapes
   - Backward pass updates weights
   - Activations are in valid ranges

3. **CLI Validation**
   - Invalid arguments raise errors
   - Valid arguments pass validation
   - Warnings are issued for problematic combinations

4. **Run Management**
   - Runs are created with correct structure
   - Config is saved correctly
   - Runs can be loaded and resumed

### Medium Priority

1. **Loss Functions**
   - Loss values are non-negative
   - Gradients are correct (numerical check)
   - Edge cases (zeros, negatives, etc.)

2. **Activation Functions**
   - Outputs are in valid ranges
   - Derivatives are correct
   - Edge cases (large values, zeros, etc.)

3. **Dataset Loading**
   - Data shapes are correct
   - Data is iterable
   - Data is properly formatted

### Low Priority

1. **Visualization**
   - Figures are created (don't test rendering)
   - File paths are correct
   - Figures are closed (memory management)

2. **Logging**
   - Logs are written (if file logging enabled)
   - Log levels are respected

## Running Tests

### Using pytest

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/data/test_dataset.py

# Run with coverage
pytest --cov=src tests/

# Run with verbose output
pytest -v
```

### Using Makefile

Add test targets to Makefile:

```makefile
.PHONY: test
test:
	@${PYTHON} -m pytest tests/

.PHONY: test-cov
test-cov:
	@${PYTHON} -m pytest --cov=src --cov-report=html tests/
```

## Test Best Practices

### 1. Use Descriptive Names

```python
# Good
def test_mse_loss_with_negative_values():
    ...

# Bad
def test_mse():
    ...
```

### 2. Test One Thing Per Test

```python
# Good
def test_loss_calculation():
    ...

def test_loss_gradient():
    ...

# Bad
def test_loss_everything():
    ...
```

### 3. Use Assertions with Messages

```python
# Good
assert loss >= 0, "Loss should be non-negative"

# Bad
assert loss >= 0
```

### 4. Test Edge Cases

```python
def test_activation_with_zero():
    ...

def test_activation_with_large_value():
    ...

def test_activation_with_negative():
    ...
```

### 5. Use Approximate Comparisons for Floats

```python
# Good
np.testing.assert_almost_equal(actual, expected, decimal=5)

# Bad
assert actual == expected  # Floats rarely equal exactly
```

### 6. Clean Up After Tests

```python
def test_something(tmpdir):
    # tmpdir is automatically cleaned up
    test_file = tmpdir / "test.txt"
    ...
```

### 7. Mock External Dependencies

```python
from unittest.mock import patch

@patch('matplotlib.pyplot.figure')
def test_graph_creation(mock_figure):
    # Test without actually creating figures
    ...
```

## Common Test Patterns

### Testing Component Registration

```python
def test_component_registration():
    from networks.loss.loss import get_available_loss_functions
    
    available = get_available_loss_functions()
    assert "mean_squared_error" in available
    assert "huber" in available
```

### Testing Factory Functions

```python
def test_get_loss_function():
    from networks.loss.loss import get_loss_function
    
    loss = get_loss_function("mean_squared_error")
    assert isinstance(loss, MeanSquaredErrorLoss)
    
    with pytest.raises(ValueError):
        get_loss_function("nonexistent")
```

### Testing Numerical Correctness

```python
def test_gradient_correctness():
    """Test gradient using finite differences"""
    loss_fn = MeanSquaredErrorLoss()
    y_true = np.array([1.0, 2.0])
    y_pred = np.array([1.1, 1.9])
    
    # Analytical gradient
    delta_analytical = loss_fn.calc_delta(y_true, y_pred)
    
    # Numerical gradient
    eps = 1e-5
    loss_plus = loss_fn.calc_loss(y_true, y_pred + eps)
    loss_minus = loss_fn.calc_loss(y_true, y_pred - eps)
    delta_numerical = (loss_plus - loss_minus) / (2 * eps)
    
    np.testing.assert_almost_equal(
        delta_analytical, delta_numerical, decimal=4
    )
```

### Testing CLI Arguments

```python
def test_cli_argument_parsing():
    from src.cli import parse_args
    
    args, command = parse_args([
        "train",
        "-n", "3",
        "-e", "10",
        "-l", "huber"
    ])
    
    assert args.n == 3
    assert args.epochs == 10
    assert args.loss == "huber"
```

## Continuous Integration

### GitHub Actions Example

```yaml
# .github/workflows/test.yml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: '3.9'
      - run: pip install -e .[dev]
      - run: pytest --cov=src tests/
```

## Summary

1. **Test components individually** - Loss functions, activations, etc.
2. **Test network operations** - Forward/backward passes
3. **Test CLI validation** - Argument parsing and validation
4. **Test integration** - End-to-end workflows
5. **Use fixtures** - For common setup
6. **Test edge cases** - Zeros, negatives, large values
7. **Use approximate comparisons** - For floating-point values
8. **Clean up** - Use tmpdir, close figures, etc.
