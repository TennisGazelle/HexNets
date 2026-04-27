# HexNets Documentation

Welcome to the HexNets documentation! This directory contains comprehensive documentation about the project architecture, usage, and development.

**Shallow index of user-facing theory docs:** [`docs/README.md`](../docs/README.md). **Paper sources** live under `docs/latex/` — agents should not edit that tree unless explicitly asked (see [`.cursor/rules/documentation-sync.mdc`](./rules/documentation-sync.mdc)).

## Documentation Files

### Core Documentation
- **[ARCHITECTURE.md](./ARCHITECTURE.md)** - Comprehensive overview of the system architecture, modularity, and design principles
- **[ROTATION_SYSTEM.md](./ROTATION_SYSTEM.md)** - Detailed explanation of how the hexagonal network rotation system works
- **[REFERENCE_FILES.md](./REFERENCE_FILES.md)** - Complete catalog of reference graph files and how to generate them
- **[QUICK_REFERENCE.md](./QUICK_REFERENCE.md)** - Quick reference guide for common tasks and CLI usage
- **[COMPONENT_DEVELOPMENT.md](./COMPONENT_DEVELOPMENT.md)** - Step-by-step guide for adding new components

### Development Guides
- **[DEVELOPMENT_PATTERNS.md](./DEVELOPMENT_PATTERNS.md)** - Important patterns, conventions, and gotchas for developers
- **[FILE_STRUCTURE.md](./FILE_STRUCTURE.md)** - Project structure, file organization, and where to find things
- **[MEMORY_MANAGEMENT.md](./MEMORY_MANAGEMENT.md)** - Memory management patterns, especially for matplotlib figures
- **[CLI_PATTERNS.md](./CLI_PATTERNS.md)** - CLI patterns, argument organization, and command structure
- **[TESTING.md](./TESTING.md)** - Testing patterns, conventions, and how to add tests

### Quick Reference
- **[AI_QUICK_INDEX.md](./AI_QUICK_INDEX.md)** - Quick reference index for AI assistants

## Quick Start

### Understanding the Modularity

HexNets uses a **plugin-based architecture** where components automatically register themselves when imported. See [ARCHITECTURE.md](./ARCHITECTURE.md) for details.

### Key Concepts

- **Base Classes** define interfaces (`BaseLoss`, `BaseActivation`, etc.)
- **Display Names** provide human-readable identifiers
- **Factory Functions** instantiate components by name
- **Discovery Functions** list all available components

**For examples and workflows, see [QUICK_REFERENCE.md](./QUICK_REFERENCE.md)**

## Running the Streamlit Web Interface

```bash
make run-streamlit
# or: streamlit run src/streamlit_app.py
```

The Streamlit app provides an interactive web interface with:
- **Network Explorer**: Interactive visualization with parameter controls
- **Rotation Comparison**: Table view of all 6 rotations (recreates the table from [ROTATION_SYSTEM.md](./ROTATION_SYSTEM.md))

**Note:** Generate reference graphs first for the Rotation Comparison tab:
```bash
hexnet ref --all
```

## Generating Reference Graphs

```bash
hexnet ref --all
```

This generates all reference graphs for n=2..8 and r=0..5. See [REFERENCE_FILES.md](./REFERENCE_FILES.md) for the complete catalog.

## Getting Help

### For Understanding the System
- **[ARCHITECTURE.md](./ARCHITECTURE.md)** - System architecture and design principles
- **[ROTATION_SYSTEM.md](./ROTATION_SYSTEM.md)** - How hexagonal rotations work
- **[FILE_STRUCTURE.md](./FILE_STRUCTURE.md)** - Project structure and organization

### For Development
- **[DEVELOPMENT_PATTERNS.md](./DEVELOPMENT_PATTERNS.md)** - Patterns, conventions, and gotchas
- **[MEMORY_MANAGEMENT.md](./MEMORY_MANAGEMENT.md)** - Memory management, especially matplotlib
- **[CLI_PATTERNS.md](./CLI_PATTERNS.md)** - CLI patterns and command structure
- **[COMPONENT_DEVELOPMENT.md](./COMPONENT_DEVELOPMENT.md)** - Adding new components
- **[TESTING.md](./TESTING.md)** - Testing patterns and conventions

### For Usage
- **[QUICK_REFERENCE.md](./QUICK_REFERENCE.md)** - Common tasks and CLI usage
- **[REFERENCE_FILES.md](./REFERENCE_FILES.md)** - Reference graph catalog

## Contributing

When adding new components:
1. Follow the patterns in existing implementations
2. Use the `display_name` parameter for registration
3. Ensure your module is imported (automatic for most cases)
4. Test with both CLI and programmatic usage
