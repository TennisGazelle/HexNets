# AI Assistant Quick Index

This is a quick reference for AI assistants working on HexNets. Use this to quickly find relevant documentation.

## Critical Patterns (Read First)

0. **Doc sync** - [.cursor/rules/documentation-sync.mdc](./rules/documentation-sync.mdc)
   - Code changes → update `README.md`, `docs/**` (not `docs/latex/` unless explicitly asked), and relevant `.cursor/**` notes concisely.

1. **Shell / venv** - [.cursor/rules/shell-venv.mdc](./rules/shell-venv.mdc)
   - Run **`make unit-test`** (and other `make` targets) or **`source .venv/bin/activate`** / **`.venv/bin/python`** before ad-hoc `pytest` / `hexnet` — avoids system Python missing `pytest` and wasted retries.

2. **Memory Management** - [MEMORY_MANAGEMENT.md](./MEMORY_MANAGEMENT.md)
   - Always close matplotlib figures with `plt.close(fig)` in finally blocks
   - Memoize network instances when generating multiple graphs

3. **Component Registration** - [ARCHITECTURE.md](./ARCHITECTURE.md#component-registration-pattern)
   - Components auto-register via `__init_subclass__` with `display_name`
   - Use discovery functions (`get_available_*()`) for CLI choices

4. **CLI Patterns** - [CLI_PATTERNS.md](./CLI_PATTERNS.md)
   - Commands follow `Command` interface
   - Use helper functions: `add_*_arguments()`, `validate_*_arguments()`

## Where to Find Things

- **File Structure**: [FILE_STRUCTURE.md](./FILE_STRUCTURE.md)
- **Component Locations**: [FILE_STRUCTURE.md#source-code-structure](./FILE_STRUCTURE.md#source-code-structure)
- **Runtime Directories**: [FILE_STRUCTURE.md#runtime-directories](./FILE_STRUCTURE.md#runtime-directories)

## Common Tasks

### Adding a New Component
1. [COMPONENT_DEVELOPMENT.md](./COMPONENT_DEVELOPMENT.md) - Step-by-step guide
2. [ARCHITECTURE.md](./ARCHITECTURE.md) - Component types and patterns
3. [DEVELOPMENT_PATTERNS.md](./DEVELOPMENT_PATTERNS.md#component-registration) - Registration pattern

### Fixing Memory Issues
1. [MEMORY_MANAGEMENT.md](./MEMORY_MANAGEMENT.md) - Matplotlib figure management
2. [DEVELOPMENT_PATTERNS.md](./DEVELOPMENT_PATTERNS.md#memory-management) - Patterns and gotchas

### Understanding CLI
1. [CLI_PATTERNS.md](./CLI_PATTERNS.md) - Command structure and patterns
2. [QUICK_REFERENCE.md](./QUICK_REFERENCE.md) - Common commands
3. [ARCHITECTURE.md](./ARCHITECTURE.md#cli-contract-and-usage) - CLI contract

### Working with Networks
1. [ARCHITECTURE.md](./ARCHITECTURE.md#core-architecture) - Base network class
2. [ROTATION_SYSTEM.md](./ROTATION_SYSTEM.md) - Hexagonal rotations
3. [DEVELOPMENT_PATTERNS.md](./DEVELOPMENT_PATTERNS.md#network-instance-memoization) - Memoization

### Running Streamlit
1. [QUICK_REFERENCE.md](./QUICK_REFERENCE.md#running-the-streamlit-web-interface) - How to launch
2. `make streamlit-run` or `streamlit run src/streamlit_app.py`
3. Requires reference graphs: `hexnet ref --all`

### Run Management
1. [CLI_PATTERNS.md](./CLI_PATTERNS.md#run-management) - Run service patterns
2. [FILE_STRUCTURE.md](./FILE_STRUCTURE.md#runs-directory) - Run directory structure
3. [DEVELOPMENT_PATTERNS.md](./DEVELOPMENT_PATTERNS.md#run-management) - Patterns

### Running unit tests
1. **`make unit-test`** from repo root (uses `.venv` automatically) — [TESTING.md](./TESTING.md), [shell-venv rule](./rules/shell-venv.mdc)
2. If not using `make`: `source .venv/bin/activate` or `.venv/bin/python -m pytest tests/`

## Common Gotchas

See [DEVELOPMENT_PATTERNS.md#common-gotchas](./DEVELOPMENT_PATTERNS.md#common-gotchas) for full list:

1. **Display names must be unique** - Last registered wins
2. **Matplotlib memory leaks** - Always close figures
3. **Run name collisions** - Check before creating
4. **Backward compatibility** - Handle old config formats
5. **Seed not set** - Set in validation, not constructor
6. **Import paths** - Use relative imports (no `src.` prefix)
7. **System Python vs venv** - `pytest` / editable install live in `.venv`; use `make` targets or activate — [shell-venv rule](./rules/shell-venv.mdc)

## Quick Lookups

### Component Types
- **Loss Functions**: `src/networks/loss/` - [ARCHITECTURE.md#loss-functions](./ARCHITECTURE.md#1-loss-functions)
- **Activations**: `src/networks/activation/` - [ARCHITECTURE.md#activation-functions](./ARCHITECTURE.md#2-activation-functions)
- **Learning Rates**: `src/networks/learning_rate/` - [ARCHITECTURE.md#learning-rate-schedules](./ARCHITECTURE.md#3-learning-rate-schedules)
- **Datasets**: `src/data/` - [ARCHITECTURE.md#datasets](./ARCHITECTURE.md#4-datasets)

### Key Files
- **CLI Entry**: `src/cli.py` - [CLI_PATTERNS.md](./CLI_PATTERNS.md)
- **Command Base**: `src/commands/command.py` - [CLI_PATTERNS.md#command-interface](./CLI_PATTERNS.md#command-interface)
- **Run Service**: `src/services/run_service/RunService.py` - [CLI_PATTERNS.md#run-management](./CLI_PATTERNS.md#run-management)
- **Network Base**: `src/networks/network.py` - [ARCHITECTURE.md#base-network-class](./ARCHITECTURE.md#base-network-class)

### Validation
- **Hex Args**: `validate_hex_only_arguments()` - [CLI_PATTERNS.md#hex-specific-arguments](./CLI_PATTERNS.md#hex-specific-arguments)
- **Global Args**: `validate_global_arguments()` - [CLI_PATTERNS.md#global-arguments](./CLI_PATTERNS.md#global-arguments)
- **Training Args**: `validate_training_arguments()` - [CLI_PATTERNS.md#training-arguments](./CLI_PATTERNS.md#training-arguments)

## Documentation Map

**Non-LaTeX theory / math snippets:** [../docs/README.md](../docs/README.md) (index of `docs/math/`, Streamlit notes). **Do not** auto-edit `docs/latex/` without an explicit maintainer request.

```
.cursor/
├── README.md                    # This directory overview
├── AI_QUICK_INDEX.md            # This file (quick reference)
│
├── Core Documentation
│   ├── ARCHITECTURE.md          # System architecture
│   ├── ROTATION_SYSTEM.md       # Hexagonal rotations
│   ├── REFERENCE_FILES.md       # Reference graph catalog
│   └── QUICK_REFERENCE.md       # CLI quick reference
│
└── Development Guides
    ├── DEVELOPMENT_PATTERNS.md  # Patterns and conventions
    ├── FILE_STRUCTURE.md        # Project structure
    ├── MEMORY_MANAGEMENT.md     # Memory patterns
    ├── CLI_PATTERNS.md          # CLI patterns
    ├── COMPONENT_DEVELOPMENT.md  # Adding components
    └── TESTING.md               # Testing guide
```

## Search Strategy

When looking for information:

1. **Quick lookup**: Use this file (AI_QUICK_INDEX.md)
2. **Pattern/convention**: [DEVELOPMENT_PATTERNS.md](./DEVELOPMENT_PATTERNS.md)
3. **Where is X?**: [FILE_STRUCTURE.md](./FILE_STRUCTURE.md)
4. **How does X work?**: [ARCHITECTURE.md](./ARCHITECTURE.md)
5. **How do I add X?**: [COMPONENT_DEVELOPMENT.md](./COMPONENT_DEVELOPMENT.md)
6. **CLI question**: [CLI_PATTERNS.md](./CLI_PATTERNS.md)
7. **Memory issue**: [MEMORY_MANAGEMENT.md](./MEMORY_MANAGEMENT.md)

## Pro Tips

1. **Always check existing patterns** - Look at similar components/commands
2. **Use helper functions** - Don't duplicate argument parsing/validation
3. **Close figures** - Always use try/finally with plt.close()
4. **Memoize networks** - When generating multiple graphs
5. **Validate early** - Check arguments before expensive operations
6. **Use project logging** - `get_logger(__name__)`, not `print()`
7. **Handle backward compatibility** - Old config formats may exist
8. **Test edge cases** - Zeros, negatives, large values, etc.

## Quick Start for New Features

1. **Understand the pattern** - Read relevant architecture docs
2. **Find similar code** - Look at existing implementations
3. **Follow conventions** - Use helper functions, patterns
4. **Test thoroughly** - Edge cases, error conditions
5. **Document** - Update relevant .cursor docs if needed

## When to Update This Documentation

Update documentation when:
- Adding new component types
- Changing CLI patterns
- Discovering new gotchas
- Adding new commands
- Changing file structure
- Finding memory issues
- Adding new patterns

Keep documentation:
- **Concise** - Prefer bullets, tables, and correct paths; avoid duplicate long explanations across `README.md`, `docs/`, and `.cursor/`.
- **Current** - Reflect actual codebase state (especially CLI names, file paths, and wired dataset types).
- **Actionable** - Show how, not just what
- **Token-aware** - Default to short sections that still answer “where / what / how to run”.
