# File Structure & Organization

This document explains the project structure, file organization, and where to find things.

## Root Directory

```
hexnets/
├── .cursor/              # AI assistant documentation (this directory)
├── docs/                 # User-facing documentation
├── figures/              # Generated figure files (created at runtime)
├── reference/            # Reference graph files (created at runtime)
├── runs/                 # Training run directories (created at runtime)
├── src/                  # Source code
├── tests/                # Test files
├── Makefile              # Development tasks
├── pyproject.toml        # Package configuration
├── setup.py              # Legacy setup (use pyproject.toml)
└── README.md             # Project overview
```

## Source Code Structure (`src/`)

```
src/
├── cli.py                # CLI entry point
├── commands/             # CLI command implementations
│   ├── command.py        # Base Command class and helpers
│   ├── train_command.py  # Training command
│   ├── reference_command.py  # `hexnet ref` — reference graph generation
│   ├── adhoc_command.py  # Ad-hoc testing
│   └── stats_conmand.py  # `hexnet stats` (filename typo; module is StatsCommand)
├── networks/              # Neural network implementations
│   ├── network.py         # Base network class
│   ├── HexagonalNetwork.py  # Hexagonal network
│   ├── MLPNetwork.py      # Multi-layer perceptron
│   ├── activation/        # Activation functions (plugin dir)
│   │   ├── activations.py # Base activation class
│   │   ├── Sigmoid.py
│   │   ├── Relu.py
│   │   ├── LeakyRelu.py
│   │   └── Linear.py
│   ├── loss/              # Loss functions (plugin dir)
│   │   ├── loss.py        # Base loss class
│   │   ├── MeanSquaredErrorLoss.py
│   │   ├── HuberLoss.py
│   │   ├── LogCoshLoss.py
│   │   └── QuantileLoss.py
│   ├── learning_rate/     # Learning rate schedules (plugin dir)
│   │   ├── learning_rate.py  # Base learning rate class
│   │   ├── ConstantLearningRate.py
│   │   └── ExponentialDecayLearningRate.py
│   ├── metrics.py         # Training metrics (loss, accuracy, r_squared, adjusted_r_squared)
├── data/                  # Dataset implementations
│   └── dataset.py         # Base dataset and implementations
├── services/              # Service modules
│   ├── figure_service/    # Figure management
│   │   ├── figure.py      # Abstract base class
│   │   ├── RefFigure.py
│   │   ├── LearningRateRefFigure.py
│   │   ├── TrainingFigure.py
│   │   └── FigureService.py
│   ├── run_service/       # Run management and persistence
│   │   └── RunService.py
│   └── logging_config/   # Logging configuration
│       └── logging_config.py
├── utils.py               # Utility functions
├── streamlit_app.py       # Streamlit web interface (see [QUICK_REFERENCE.md](./QUICK_REFERENCE.md#running-the-streamlit-web-interface))
└── obsolete/              # Old/unused code (for reference)
```

## Key Files Explained

### Entry Points

- **`src/cli.py`**: Main CLI entry point. Registers commands and parses arguments.
- **`pyproject.toml`**: Defines `hexnet` CLI command pointing to `cli:main`

### Command System

- **`src/commands/command.py`**: Base `Command` class and argument helpers
  - `add_hex_only_arguments()` - Hex-specific args (`-n`, `-r`)
  - `add_global_arguments()` - Model-agnostic args (`-m`, `-a`, `-l`, `-s`)
  - `add_training_arguments()` - Training args (`-lr`, `-e`, `-p`, `-t`, `-ds`)
  - Validation functions for each group

### Network Implementations

- **`src/networks/network.py`**: Abstract base class defining network interface
- **`src/networks/HexagonalNetwork.py`**: Hexagonal network with rotation support
- **`src/networks/MLPNetwork.py`**: Standard MLP implementation

### Component Plugins

Component directories (`activation/`, `loss/`, `learning_rate/`) contain base class files and implementations. **For registration patterns, see [ARCHITECTURE.md](./ARCHITECTURE.md#component-registration-pattern)**

### Run Management

- **`src/services/run_service/RunService.py`**: `RunService` class handles run creation, loading, and persistence. **For details, see [CLI_PATTERNS.md](./CLI_PATTERNS.md#run-management)**

### Training Metrics

- **`src/networks/metrics.py`**: `Metrics` class tracks training progress
  - Tracks per-epoch: `loss`, `accuracy`, `r_squared`, `adjusted_r_squared`
  - Stores intermediate values for R² calculation: `ss_res_sum`, `y_sum`, `y2_sum`, `count`
  - Methods: `add_metric()`, `calc_accuracy_r2()`, `tally_accurcy_r2()`
  - Serialized to `training_metrics.json` in run directories

### Utilities

- **`src/utils.py`**: Helper functions
  - `table_print()` - Pretty table printing
  - `get_json_file_contents()` - JSON file reading
  - `Colors` - Terminal color codes

- **`src/services/logging_config/logging_config.py`**: Logging setup
  - `setup_logging()` - Configure logging
  - `get_logger()` - Get module logger

- **`src/services/figure_service/`**: Figure management
  - `Figure` - Abstract base class for figures
  - `RefFigure` - Reference figure implementation
  - `LearningRateRefFigure` - Learning rate visualization
  - `TrainingFigure` - Training progress visualization
  - `FigureService` - Service class for managing figures

## Runtime Directories

### `runs/` Directory

Created automatically. Structure:

```
runs/
  YYYY-MM-DD_HH-MM_<uuid>/
    config.json              # Hyperparameters
    manifest.json            # Hashes and metadata
    training_metrics.json    # Training history (loss, accuracy, r_squared, adjusted_r_squared arrays)
    model.pkl               # Saved weights
    plots/                  # Generated figures
      *.png
```

**Naming:**
- Auto-generated: `YYYY-MM-DD_HH-MM_<6-char-uuid>`
- Custom: Use `-rn/--run_name` flag

### `reference/` Directory

Created automatically. Contains reference graphs:

```
reference/
  hexnet_n2_r0_structure.png
  hexnet_n2_r0_Activation_Structure.png
  hexnet_n2_r0_Weight_Matrix.png
  hexnet_n2_multi_activation.png
  ...
```

**Naming pattern:** `hexnet_n{n}_r{r}_{type}.png`

### `figures/` Directory

Legacy directory for ad-hoc figure generation. Not used by main commands.

## Test Structure

```
tests/
  data/
    test_dataset.py    # Dataset tests
```

**Note:** Test coverage is minimal. Tests should be added for:
- Component registration
- Network forward/backward passes
- CLI argument validation
- Run save/load

## Obsolete Directory

`src/obsolete/` contains old implementations for reference:
- `hexnet_with_activation_base_class.py` - Old activation pattern
- `mlp.py` - Old MLP implementation
- `mnist.py` - MNIST dataset (not used)
- `some_cache.py` - Caching experiments
- `hexnet-class.ipynb` - Jupyter notebook experiments

**Do not modify these files.** They're kept for historical reference.

## Documentation Structure

### `.cursor/` Directory

AI assistant documentation (this directory):

- **README.md** - Overview and navigation
- **ARCHITECTURE.md** - System architecture
- **ROTATION_SYSTEM.md** - Hexagonal rotation explanation
- **REFERENCE_FILES.md** - Reference graph catalog
- **QUICK_REFERENCE.md** - CLI quick reference
- **COMPONENT_DEVELOPMENT.md** - Adding new components
- **DEVELOPMENT_PATTERNS.md** - Patterns and conventions (this file's sibling)
- **FILE_STRUCTURE.md** - This file

### `docs/` Directory

User-facing documentation:
- `dataset_examples.md` - Dataset usage examples

## Import Patterns

### Within Package

Use relative imports (no `src.` prefix):

```python
from networks.HexagonalNetwork import HexagonalNeuralNetwork
from commands.command import Command
from data.dataset import IdentityDataset
from services.figure_service import FigureService
from services.run_service import RunService
from services.logging_config import get_logger
```

### Entry Point

The CLI entry point is `cli:main` (not `src.cli:main`) because the package root is `src/`.

## File Naming Conventions

- **Classes**: `PascalCase` (e.g., `HexagonalNetwork.py`)
- **Display names**: `snake_case` (e.g., `"mean_squared_error"`)
- **CLI commands**: `snake_case` (e.g., `train_command.py`)
- **Test files**: `test_*.py`

## Where to Add Things

### New Component

**See [COMPONENT_DEVELOPMENT.md](./COMPONENT_DEVELOPMENT.md) for step-by-step guide**

1. Create file in appropriate plugin directory
2. Inherit from base class with `display_name`
3. Ensure module is imported

### New Network Type

1. Create `src/networks/MyNetwork.py`
2. Inherit from `BaseNeuralNetwork`
3. Implement all abstract methods
4. Add handling in `RunService` if needed for CLI

### New Command

1. Create `src/commands/my_command.py`
2. Inherit from `Command`
3. Implement required methods
4. Register in `src/cli.py`

### New Dataset

1. Add to `src/data/dataset.py` or create new file
2. Inherit from `BaseDataset` with `display_name`
3. Update `get_dataset()` helper in `src/commands/command.py`

## Build Artifacts

### `.venv/`

Virtual environment (created by `make install`). Contains:
- Installed packages
- CLI entry point (`hexnet` command)

### `src/*.egg-info/`

Package metadata (created during install). Can be safely deleted.

### `__pycache__/`

Python bytecode cache. Can be safely deleted.

## Makefile Targets

See `Makefile` for available targets:
- `make install` - Set up environment
- `make clean-ref` - Remove reference graphs
- `make clean-figures` - Remove figure files
- `make clean-runs` - Remove run directories
- `make clean-all` - Remove everything

## Important Paths

- **CLI entry point**: `src/cli.py:main()`
- **Package root**: `src/` (not project root)
- **Runs directory**: `runs/` (relative to project root)
- **Reference directory**: `reference/` (relative to project root)
- **Figures directory**: `figures/` (relative to project root)
