# Development Patterns & Conventions

This document outlines important patterns, conventions, and gotchas that developers should be aware of when working on HexNets.

## Table of Contents
- [Memory Management](#memory-management)
- [CLI Command Pattern](#cli-command-pattern)
- [Component Registration](#component-registration)
- [Run Management](#run-management)
- [Figure Generation](#figure-generation)
- [Error Handling](#error-handling)
- [Logging](#logging)
- [Common Gotchas](#common-gotchas)

## Memory Management

**For detailed memory management patterns, see [MEMORY_MANAGEMENT.md](./MEMORY_MANAGEMENT.md)**

**CRITICAL Patterns:**
- Always close matplotlib figures with `plt.close(fig)` in finally blocks
- Memoize network instances when generating multiple graphs (create once, reuse many times)

## CLI Command Pattern

**For detailed CLI patterns, see [CLI_PATTERNS.md](./CLI_PATTERNS.md)**

**Key Points:**
- Commands implement `Command` interface with `name()`, `help()`, `configure_parser()`, `validate_args()`, `invoke()`
- Use helper functions: `add_hex_only_arguments()`, `add_global_arguments()`, `add_training_arguments()`
- Validation order: type/range → compatibility warnings → business logic

## Component Registration

**For detailed component registration patterns, see [ARCHITECTURE.md](./ARCHITECTURE.md#component-registration-pattern)**

**Key Points:**
- Components auto-register via `__init_subclass__` with `display_name` parameter
- Registration happens at class definition time (import time)
- Discovery functions (`get_available_*()`) enable CLI dynamic choices
- Factory functions (`get_*_function()`) instantiate by name

## Run Management

**For detailed run management patterns, see [CLI_PATTERNS.md](./CLI_PATTERNS.md#run-management)**

**Key Points:**
- `RunService` handles both initialization and loading
- Run names must be unique (validated)
- Cannot specify both `run_dir` and `run_name`
- Handle backward compatibility when loading old config formats

## Figure Generation

### Reference vs Training Figures

- **Reference figures**: Structure-based, saved to `reference/` directory
- **Training figures**: Run-specific, saved to `runs/<run_name>/plots/`

### Figure Service

`FigureService` provides a unified interface for figure management:

```python
from figure_service import FigureService

service = FigureService()
service.set_figures_path(run.get_figures_path())
fig = service.init_training_figure(...)
```

**Note:** Currently only used for training figures. Reference figures use direct matplotlib.

### Graph Methods

Network graph methods return `(filepath, figure)` tuple:

```python
def graph_structure(...) -> Tuple[str, plt.Figure]:
    # ... create figure ...
    return str(full_path), fig
```

**Why:** Allows caller to manage figure lifecycle if needed.

## Error Handling

### Validation Errors

Use `ValueError` for invalid arguments:

```python
if args.n < 2:
    raise ValueError("Number of input nodes must be at least 2")
```

### Warnings vs Errors

- **Errors**: Invalid arguments, missing files, impossible states
- **Warnings**: Problematic but valid combinations (e.g., sigmoid with regression)

```python
# Warning: Valid but problematic
logger.warning("Sigmoid output bounds predictions to (0, 1)...")

# Error: Invalid
raise ValueError("Invalid rotation: must be 0-5")
```

### Exception Handling in Loops

When generating multiple items, catch exceptions per item:

```python
for r in range(6):
    try:
        nets[r].graph_structure(...)
    except Exception as e:
        print(f"Error for r={r}: {e}")
        logger.exception(f"Error generating graph for r={r}")
        continue  # Don't stop entire process
```

**See:** `src/commands/reference_command.py` - `_generate_for_n` method.

## Logging

### Logger Setup

Always use the project's logging config:

```python
from logging_config import get_logger

logger = get_logger(__name__)
```

**Why:** Ensures consistent formatting and level management.

### Log Levels

- **DEBUG**: Detailed diagnostic information
- **INFO**: General informational messages
- **WARNING**: Problematic but valid situations
- **ERROR**: Error conditions

### Third-Party Logging

Third-party loggers are set to WARNING to reduce noise:

```python
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("PIL").setLevel(logging.WARNING)
```

**See:** `src/logging_config.py`.

## Common Gotchas

### 1. Display Names Must Be Unique

If two components have the same `display_name`, the last one registered wins (silently overwrites).

**Fix:** Use descriptive, unique names.

### 2. Component Not Discoverable

If a component isn't showing up in CLI choices:

1. Check `display_name` is provided
2. Ensure module is imported (check `__init__.py`)
3. Verify class inherits from base class correctly

### 3. Matplotlib Memory Leaks

**Symptom:** `RuntimeWarning: More than 20 figures have been opened`

**Fix:** Always use `plt.close(fig)` in finally blocks.

### 4. Run Name Collisions

Run names must be unique. Check before creating:

```python
if (RunService.runs_dir / args.run_name).exists():
    raise ValueError(f"Run named '{args.run_name}' already exists")
```

### 5. Backward Compatibility

When changing config format, handle old formats:

```python
# Old format: learning_rate was a float
# New format: learning_rate is a string
if isinstance(learning_rate_config, (int, float)):
    learning_rate_config = "constant"
```

### 6. Seed Not Set

Random seed must be set in validation, not in network constructor:

```python
def validate_training_arguments(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
```

**Why:** Ensures reproducibility before any random operations.

### 7. Figure Path Not Created

Always create parent directories:

```python
parent_dir = pathlib.Path(output_dir) if output_dir else pathlib.Path("figures")
parent_dir.mkdir(parents=True, exist_ok=True)
full_path = parent_dir / filename
```

### 8. Network State Not Saved

Training metrics must be explicitly saved:

```python
run.set_training_metrics(net.get_training_metrics())
run.output_run_files()
```

**See:** `src/commands/train_command.py`.

### 9. CLI Entry Point

The CLI entry point is `cli:main`, not `src.cli:main`:

```python
# pyproject.toml
[project.scripts]
hexnet = "cli:main"  # Note: no "src."
```

**Why:** Package is installed with `src/` as the root.

### 10. Import Paths

Use relative imports within package:

```python
# Good
from networks.HexagonalNetwork import HexagonalNeuralNetwork
from commands.command import Command

# Bad (breaks when installed)
from src.networks.HexagonalNetwork import HexagonalNeuralNetwork
```

## Best Practices Summary

1. **Always close matplotlib figures** - Use try/finally
2. **Memoize expensive objects** - Networks, datasets, etc.
3. **Validate early** - Check arguments before expensive operations
4. **Handle exceptions gracefully** - Don't crash entire batch operations
5. **Use project logging** - Don't use `print()` for errors
6. **Document display names** - They're the public API
7. **Test backward compatibility** - Handle old config formats
8. **Create directories explicitly** - Don't assume they exist
9. **Use helper functions** - `add_*_arguments()`, `validate_*_arguments()`
10. **Follow naming conventions** - `PascalCase` for classes, `snake_case` for display names
