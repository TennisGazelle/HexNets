# CLI Patterns & Usage

This document explains CLI patterns, argument organization, and command structure.

## Command Registration

Commands are registered in `src/cli.py`:

```python
commands = [
    ReferenceCommand(),
    AdhocCommand(),
    TrainCommand(),
    StatsCommand()
]

for command in commands:
    subparser = subparsers.add_parser(command.name(), help=command.help())
    subparser.set_defaults(command=command)
    command.configure_parser(subparser)
```

**Key Points:**
- Commands are instantiated once (singleton-like)
- Each command configures its own subparser
- Command instance is stored in `args.command`

## Command Interface

All commands implement the `Command` abstract base class:

```python
class Command(ABC):
    @abstractmethod
    def name(self) -> str:
        """Return command name (e.g., 'train')"""
    
    @abstractmethod
    def help(self) -> str:
        """Return help text"""
    
    @abstractmethod
    def configure_parser(self, parser: ArgumentParser):
        """Add arguments to parser"""
    
    @abstractmethod
    def validate_args(self, args: Namespace):
        """Validate arguments"""
    
    @abstractmethod
    def invoke(self, args: Namespace):
        """Execute command"""
```

### Command Execution Flow

```python
def __call__(self, args: Namespace):
    print_header()           # ASCII art header
    self.validate_args(args) # Validate before execution
    self.invoke(args)        # Execute command
```

## Argument Organization

Arguments are organized into logical groups using helper functions:

### Hex-Specific Arguments

```python
add_hex_only_arguments(parser)
```

Adds:
- `-n, --num_dims`: Input/output dimensions (default: 3)
- `-r, --rotation`: Hex rotation 0-5 (default: 0)

**Used by:** Commands that work with hexagonal networks

### Global Arguments

```python
add_global_arguments(parser)
```

Adds:
- `-m, --model`: Model type: `hex` or `mlp` (default: `hex`)
- `-a, --activation`: Activation function (choices from registry)
- `-l, --loss`: Loss function (choices from registry)
- `-s, --seed`: Random seed (default: 42)

**Used by:** All commands that create networks

### Training Arguments

```python
add_training_arguments(parser)
```

Adds:
- `-lr, --learning-rate`: Learning rate schedule (default: `constant`)
- `-e, --epochs`: Number of epochs (default: 100)
- `-p, --pause`: Pause between epochs for animation (default: 0.05)
- `-t, --type`: Dataset type: `identity` or `linear` (default: `identity`)
- `-ds, --dataset-size`: Number of samples (default: 250)
- `--dry-run`: Preview without creating run

**Used by:** Commands that train networks

## Validation Flow

### Validation Order

1. **Type/range validation** - Check argument types and ranges
2. **Compatibility warnings** - Warn about problematic combinations
3. **Business logic validation** - Check run names, directories, etc.

### Validation Functions

```python
validate_hex_only_arguments(args)    # Check -n, -r
validate_global_arguments(args)      # Check -m, -a, -l, -s
validate_training_arguments(args)    # Check training args, set seed
```

### Compatibility Warnings

`validate_global_arguments()` warns about problematic combinations:

- **Sigmoid + Regression**: Outputs bounded to [0,1]
- **ReLU + Regression**: Outputs bounded to [0,∞)
- **LeakyReLU + Regression**: Asymmetric scaling
- **Quantile loss**: Missing quantile parameter
- **Huber loss**: Missing delta parameter

**These are warnings, not errors** - Command still executes.

## Available Commands

### `train` - Training Command

**Purpose:** Train a network and save results to a run directory

**Arguments:**
- All hex-specific arguments
- All global arguments
- All training arguments
- `-rd, --run-dir`: Load existing run
- `-rn, --run_name`: Custom run name

**Example:**
```bash
hexnet train -n 3 -r 1 -e 100 -l huber -a relu -rn my_experiment
```

**Behavior:**
- Creates new run (or loads existing if `-rd` provided)
- Trains network
- Saves config, metrics, model, plots
- Prints run directory path

### `ref` - Reference Graph Generation

**Purpose:** Generate reference graphs (structure visualizations)

**Arguments:**
- Hex-specific arguments (`-n`, `-r`)
- Global arguments (`-m`, `-a`, `-l`)
- `-g, --graph`: Graph type (default: `structure_matplotlib`)
- `--detail`: Subtitle for graph
- `--all`: Generate all graphs for n=2..8, r=0..5

**Graph Types:**
- `structure_dot`: Graphviz DOT format
- `structure_matplotlib`: Matplotlib visualization
- `activation`: Activation matrix
- `weight`: Weight matrix
- `multi_activation`: Multi-rotation overlay
- `layer_indices_terminal`: Terminal output of layer indices

**Example:**
```bash
# Single graph
hexnet ref -n 3 -r 1 -g structure_matplotlib

# All reference graphs
hexnet ref --all
```

**Note:** When `--all` is used, `-n` and `-r` are ignored (with warning).

### `adhoc` - Ad-Hoc Testing

**Purpose:** Quick testing and experimentation

**Arguments:**
- Hex-specific arguments
- Global arguments
- Training arguments
- Custom arguments for ad-hoc use

**Example:**
```bash
hexnet adhoc -n 3 -lr 0.001 -t linear -e 200
```

**Behavior:**
- Doesn't create a run directory
- Useful for quick experiments
- Outputs to console/figures directory

### `stats` - Run Statistics

**Purpose:** Display statistics about training runs

**Arguments:**
- `-rd, --run-dir`: Run directory to analyze

**Example:**
```bash
hexnet stats -rd runs/2025-01-01_12-00_abc123
```

## Argument Patterns

### Mutually Exclusive Arguments

Some arguments are mutually exclusive:

```python
# train command
if args.run_name and args.run_dir:
    raise ValueError("Cannot define both run_name and run_dir")
```

### Conditional Arguments

Some arguments only make sense in certain contexts:

```python
# ref command
if args.generate_all:
    # Warn if -n or -r provided (they'll be ignored)
    logger.warning("--all flag ignores -n and -r arguments")
```

### Default Values

Defaults are set in argument definitions:

```python
parser.add_argument(
    "-n", "--num_dims",
    type=int,
    default=3,  # Default value
    help="...",
)
```

**Note:** Defaults should be sensible for most use cases.

## Discovery Functions

Commands use discovery functions to populate choices:

```python
parser.add_argument(
    "-a", "--activation",
    choices=get_available_activation_functions(),  # Dynamic!
    ...
)
```

**Benefits:**
- Automatically includes new components
- No manual updates needed
- Consistent across commands

## Error Handling

### Validation Errors

Raise `ValueError` for invalid arguments:

```python
if args.n < 2:
    raise ValueError("Number of input nodes must be at least 2")
```

### Warnings

Use `logger.warning()` for problematic but valid combinations:

```python
if act == "sigmoid" and loss in {"mse", "logcosh"}:
    logger.warning("Sigmoid output bounds predictions to (0, 1)...")
```

### Exception Handling

Commands should handle exceptions gracefully:

```python
def invoke(self, args):
    try:
        # ... command logic ...
    except Exception as e:
        logger.error(f"Command failed: {e}")
        logger.exception("Full traceback:")
        raise  # Re-raise or exit gracefully
```

## Command-Specific Patterns

### Train Command

**Pattern:** Create/load run → Train → Save

```python
def invoke(self, args):
    # Create/load run
    run = RunService(args)
    net = run.net
    
    # Train
    data = get_dataset(...)
    net.train(data, epochs=args.epochs)
    
    # Save
    run.set_training_metrics(net.get_training_metrics())
    run.output_run_files()
```

### Reference Command

**Pattern:** Generate graphs → Save to reference/

```python
def invoke(self, args):
    if args.generate_all:
        self._generate_all_refs(figures_dir)
    else:
        # Single graph
        net = create_network(args)
        net.graph_structure(...)
```

**Optimization:** Memoize networks when generating multiple graphs.

### Adhoc Command

**Pattern:** Quick test → No persistence

```python
def invoke(self, args):
    net = create_network(args)
    data = get_dataset(...)
    net.train_animated(data, epochs=args.epochs)
    # No run directory created
```

## Best Practices

1. **Use helper functions** - `add_*_arguments()`, `validate_*_arguments()`
2. **Validate early** - Check arguments before expensive operations
3. **Provide helpful errors** - Clear error messages with suggestions
4. **Use warnings for compatibility** - Don't block valid but problematic combinations
5. **Discover components dynamically** - Use registry functions for choices
6. **Handle exceptions gracefully** - Don't crash on user errors
7. **Print helpful output** - Show what's happening, paths, etc.
8. **Follow naming conventions** - Consistent argument names across commands

## Common Patterns

### Creating a Network

```python
activation = get_activation_function(args.activation)
loss = get_loss_function(args.loss)

if args.model == "hex":
    net = HexagonalNeuralNetwork(
        n=args.n,
        r=args.rotation,
        activation=activation,
        loss=loss,
        learning_rate=args.learning_rate,
    )
elif args.model == "mlp":
    net = MLPNetwork(
        input_dim=args.n,
        output_dim=args.n,
        hidden_dims=[4, 5, 4],
        activation=activation,
        loss=loss,
        learning_rate=args.learning_rate,
    )
```

### Getting a Dataset

```python
data = get_dataset(
    n=args.n,
    train_samples=args.dataset_size,
    type=args.type,
    scale=2.0 if args.type == "linear" else 1.0
)
```

### Setting Random Seed

```python
# Done in validate_training_arguments()
random.seed(args.seed)
np.random.seed(args.seed)
```

## Summary

- Commands follow a consistent interface
- Arguments are organized into logical groups
- Validation happens before execution
- Discovery functions enable dynamic choices
- Errors are clear, warnings are informative
- Commands handle exceptions gracefully
