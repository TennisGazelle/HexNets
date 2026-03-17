# HexNets Architecture Documentation

## Overview

HexNets is a neural network framework featuring hexagonal network architectures with a highly modular design. The framework allows you to easily swap out components like loss functions, activation functions, learning rate schedules, and datasets through a plugin-like architecture.

## Core Architecture

### Base Network Class

The foundation of the framework is `BaseNeuralNetwork` (in `src/networks/network.py`), which defines the abstract interface that all network implementations must follow. The base class enforces:

- **Abstract methods** that must be implemented:
  - `forward(x)` - Forward pass through the network
  - `backward(activations, target)` - Backward pass for gradient computation
  - `train(data, epochs)` - Training loop
  - `test(x)` - Inference
  - `save(filepath)` / `load(filepath)` - Model persistence
  - `graph_weights()` / `graph_structure()` - Visualization

- **Composable properties** injected via constructor:
  - `activation: BaseActivation` - Activation function
  - `loss: BaseLoss` - Loss function
  - `learning_rate_fn: BaseLearningRate` - Learning rate schedule

### Network Implementations

Currently, the framework includes:

1. **HexagonalNeuralNetwork** (`src/networks/HexagonalNetwork.py`)
   - Implements a hexagonal topology with 6 rotational views
   - Parameters: `n` (input/output dimensions), `r` (rotation 0-5)
   - Uses a global weight matrix shared across all rotations
   - **See [ROTATION_SYSTEM.md](./ROTATION_SYSTEM.md) for detailed explanation of how rotations work**

2. **MLPNetwork** (`src/networks/MLPNetwork.py`)
   - Standard multi-layer perceptron
   - Parameters: `input_dim`, `hidden_dims`, `output_dim`

## Modular Component System

The framework uses a **plugin-based registration system** where components automatically register themselves when imported. This allows for easy extensibility without modifying core code.

### Component Registration Pattern

Each component type follows this pattern:

1. **Base abstract class** defines the interface
2. **Registry dictionary** stores all implementations
3. **Automatic registration** via `__init_subclass__` hook
4. **Discovery functions** list available implementations
5. **Factory functions** instantiate components by name

### Component Types

#### 1. Loss Functions (`src/networks/loss/`)

**Base Class:** `BaseLoss` (`src/networks/loss/loss.py`)

**Required Methods:**
- `calc_loss(y_true, y_pred)` - Compute loss value
- `calc_delta(y_true, y_pred)` - Compute gradient delta for backprop

**Available Implementations:**
- `MeanSquaredErrorLoss` (display_name: `"mean_squared_error"`)
- `HuberLoss` (display_name: `"huber"`)
- `LogCoshLoss` (display_name: `"logcosh"`)
- `QuantileLoss` (display_name: `"quantile"`)

**For adding new loss functions, see [COMPONENT_DEVELOPMENT.md](./COMPONENT_DEVELOPMENT.md#adding-a-new-loss-function)**

#### 2. Activation Functions (`src/networks/activation/`)

**Base Class:** `BaseActivation` (`src/networks/activation/activations.py`)

**Required Methods:**
- `activate(x)` - Apply activation function
- `deactivate(x)` - Compute derivative (for backprop)

**Available Implementations:**
- `Sigmoid` (display_name: `"sigmoid"`)
- `ReLU` (display_name: `"relu"`)
- `LeakyReLU` (display_name: `"leakyrelu"`)
- `Linear` (display_name: `"linear"`)

**For adding new activations, see [COMPONENT_DEVELOPMENT.md](./COMPONENT_DEVELOPMENT.md#adding-a-new-activation-function)**

#### 3. Learning Rate Schedules (`src/networks/learning_rate/`)

**Base Class:** `BaseLearningRate` (`src/networks/learning_rate/learning_rate.py`)

**Required Methods:**
- `rate_at_iteration(iteration: int) -> float` - Get learning rate for given iteration

**Available Implementations:**
- `ConstantLearningRate` (display_name: `"constant"`)

**For adding new learning rate schedules, see [COMPONENT_DEVELOPMENT.md](./COMPONENT_DEVELOPMENT.md#adding-a-new-learning-rate-schedule)**

#### 4. Datasets (`src/data/`)

**Base Class:** `BaseDataset` (`src/data/dataset.py`)

**Required Methods:**
- `load_data() -> bool` - Load/prepare the dataset

**Required Properties:**
- `data: dict` with keys `"X"` (inputs) and `"Y"` (targets)
- Must be iterable (implements `__iter__` and `__getitem__`)

**Available Implementations:**
- `IdentityDataset` (display_name: `"identity"`) - Maps input to itself
- `LinearScaleDataset` (display_name: `"linear_scale"`) - Scales input by a factor
- `DiagonalScaleDataset` (display_name: `"diagonal_scale"`) - Scales each dimension differently

**For adding new datasets, see [COMPONENT_DEVELOPMENT.md](./COMPONENT_DEVELOPMENT.md#adding-a-new-dataset)**

## CLI Contract and Usage

**For detailed CLI documentation, see [CLI_PATTERNS.md](./CLI_PATTERNS.md)**

The CLI uses a command pattern where each command implements the `Command` interface. Available commands:
- `train` - Train a network and save results
- `adhoc` - Ad-hoc testing and experimentation  
- `ref` - Generate reference graphs (use `--all` for all graphs)
- `stats` - Display statistics about runs

Arguments are organized into groups: hex-specific (`-n`, `-r`), global (`-m`, `-a`, `-l`, `-s`), and training (`-lr`, `-e`, `-p`, `-t`, `-ds`).

**For iteration examples, see [QUICK_REFERENCE.md](./QUICK_REFERENCE.md)**

**For detailed CLI usage, run management, and validation, see [CLI_PATTERNS.md](./CLI_PATTERNS.md)**

## Design Principles

1. **Plugin Architecture** - Components auto-register on import
2. **Duck Typing** - Components work via interface, not inheritance hierarchy
3. **Display Names** - Human-readable names separate from class names
4. **Factory Pattern** - `get_*_function()` factories for instantiation
5. **Validation** - CLI validates arguments and warns about incompatibilities
6. **Persistence** - Runs save complete configuration for reproducibility

## Extension Points

To extend the framework:

1. **Add a new component type:**
   - Create base class with `__init_subclass__` registration
   - Create registry dictionary
   - Add discovery and factory functions
   - Update CLI argument parsers

2. **Add a new network type:**
   - Inherit from `BaseNeuralNetwork`
   - Implement all abstract methods
   - Add CLI argument handling in `RunService`

3. **Add a new command:**
   - Create class inheriting from `Command`
   - Implement all required methods
   - Register in `src/cli.py`
