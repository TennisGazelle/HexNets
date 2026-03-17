# Memory Management Guide

This document covers memory management patterns, especially for matplotlib figures and network instances.

## Matplotlib Figure Management

### The Problem

Matplotlib keeps all figures in memory until explicitly closed. When generating many graphs (e.g., `ref --all` generates 133+ figures), this can quickly consume gigabytes of memory and trigger warnings:

```
RuntimeWarning: More than 20 figures have been opened. 
Figures created through the pyplot interface are retained until 
explicitly closed and may consume too much memory.
```

### The Solution

**Always close figures after use:**

```python
fig = plt.figure(figsize=(7, 7))
try:
    # ... plotting code ...
    plt.savefig(path)
    plt.show()
finally:
    plt.close(fig)  # CRITICAL: Always close!
```

### Pattern in HexagonalNetwork

All graph methods in `HexagonalNetwork` follow this pattern:

```python
def graph_weights(self, ...):
    fig = plt.figure(figsize=(7, 7))
    try:
        # Create plot
        plt.imshow(...)
        plt.savefig(full_path)
        plt.show()
    finally:
        plt.close(fig)  # Ensures cleanup even on error
    return str(full_path), fig
```

**Methods using this pattern:**
- `graph_weights()`
- `_graph_multi_activation()`
- `_graph_hex()` (structure graphs)

### When to Close

Close figures:
- ✅ After saving to file
- ✅ After displaying (if needed)
- ✅ In finally blocks to ensure cleanup on errors
- ✅ Before returning from function

Don't close:
- ❌ If caller needs to keep figure alive (rare)
- ❌ If using figure service (it manages lifecycle)

### Figure Service

`FigureService` manages training figures. It doesn't automatically close figures, but training figures are typically long-lived (updated during training).

For reference graphs, use direct matplotlib with explicit closing.

## Network Instance Memoization

### The Problem

When generating multiple graphs for the same network configuration, creating new network instances is wasteful:

```python
# BAD: Creates 25 network instances
for r in range(6):
    net = HexagonalNeuralNetwork(n=2, r=r, ...)
    net.graph_structure(...)
    net.graph_weights(...)
    # ... more operations ...
```

### The Solution

**Create networks once, reuse for all operations:**

```python
# GOOD: Creates 6 network instances, reuses for all operations
nets = {}
for r in range(6):
    nets[r] = HexagonalNeuralNetwork(n=2, r=r, ...)

# Reuse for all graph operations
for r in range(6):
    nets[r].graph_structure(...)
    nets[r].graph_weights(activation_only=True, ...)
    nets[r].graph_weights(activation_only=False, ...)
```

### Pattern in ReferenceCommand

The `_generate_for_n` method uses this pattern:

```python
def _generate_for_n(self, n, figures_dir, activation_function, loss_function):
    # Create networks once
    nets = {}
    for r in range(6):
        nets[r] = HexagonalNeuralNetwork(
            n=n, r=r,
            activation=activation_function,
            loss=loss_function,
            ...
        )
    
    # Reuse for all operations
    for r in range(6):
        nets[r]._print_indices(r)
    
    nets[0]._graph_multi_activation(...)
    
    for r in range(6):
        nets[r].graph_structure(...)
        nets[r].graph_weights(activation_only=True, ...)
        nets[r].graph_weights(activation_only=False, ...)
```

**Benefits:**
- Reduces network creation from 25 to 6 instances
- Faster execution (no redundant initialization)
- Lower memory usage

## Memory Leak Detection

### Symptoms

- `RuntimeWarning` about too many open figures
- Memory usage growing over time
- System becoming slow after many operations

### Debugging

Check open figures:

```python
import matplotlib.pyplot as plt

print(f"Open figures: {len(plt.get_fignums())}")
```

Close all figures:

```python
plt.close('all')  # Emergency cleanup
```

### Prevention Checklist

- [ ] All `plt.figure()` calls have corresponding `plt.close()`
- [ ] Use try/finally blocks for figure cleanup
- [ ] Networks are memoized when reused
- [ ] Large arrays are deleted when no longer needed
- [ ] Generators used for large datasets (if applicable)

## Best Practices

### 1. Always Use Try/Finally

```python
fig = plt.figure()
try:
    # ... plotting ...
finally:
    plt.close(fig)  # Always executes
```

### 2. Memoize Expensive Objects

```python
# Create once
networks = {r: create_network(r) for r in range(6)}

# Reuse many times
for operation in operations:
    for r in range(6):
        networks[r].do_operation(operation)
```

### 3. Close Figures Immediately After Use

```python
# Good: Close right after use
fig = create_figure()
save_figure(fig)
plt.close(fig)

# Bad: Keep figure alive unnecessarily
fig = create_figure()
save_figure(fig)
# ... many lines later ...
plt.close(fig)  # Too late!
```

### 4. Use Context Managers (Future Enhancement)

Consider implementing a context manager:

```python
@contextmanager
def managed_figure(figsize=(7, 7)):
    fig = plt.figure(figsize=figsize)
    try:
        yield fig
    finally:
        plt.close(fig)

# Usage:
with managed_figure() as fig:
    plt.plot(...)
    plt.savefig(...)
```

### 5. Batch Operations

When generating many figures, process in batches:

```python
BATCH_SIZE = 10
for i in range(0, len(items), BATCH_SIZE):
    batch = items[i:i+BATCH_SIZE]
    for item in batch:
        generate_figure(item)
    # All figures from batch are closed
    # Memory is freed before next batch
```

## Performance Considerations

### Memory vs Speed Trade-offs

- **Memoization**: Uses more memory, but faster (reuse objects)
- **Lazy creation**: Uses less memory, but slower (create on demand)

For reference graph generation:
- **Memoize networks** (created once, reused many times)
- **Don't memoize figures** (create, use, close immediately)

### Large Networks

For very large networks (high `n` values):
- Weight matrices can be large (O(n²) memory)
- Consider processing one rotation at a time
- Close figures immediately after saving

### Dataset Memory

Datasets are typically small (hundreds of samples), but:
- Use generators for very large datasets
- Delete datasets after use if memory is tight
- Consider streaming for large-scale training

## Common Mistakes

### 1. Forgetting to Close Figures

```python
# BAD
fig = plt.figure()
plt.plot(...)
plt.savefig(...)
# Missing plt.close(fig)!

# GOOD
fig = plt.figure()
try:
    plt.plot(...)
    plt.savefig(...)
finally:
    plt.close(fig)
```

### 2. Creating Networks in Loops

```python
# BAD: Creates many instances
for operation in operations:
    net = HexagonalNeuralNetwork(...)
    net.do_operation(operation)

# GOOD: Create once, reuse
net = HexagonalNeuralNetwork(...)
for operation in operations:
    net.do_operation(operation)
```

### 3. Keeping References to Figures

```python
# BAD: Keeps figure in memory
figures = []
for i in range(100):
    fig = plt.figure()
    plt.plot(...)
    figures.append(fig)  # Never closed!

# GOOD: Close immediately
for i in range(100):
    fig = plt.figure()
    try:
        plt.plot(...)
        plt.savefig(f"plot_{i}.png")
    finally:
        plt.close(fig)
```

## Summary

1. **Always close matplotlib figures** - Use try/finally
2. **Memoize network instances** - Create once, reuse many times
3. **Close figures immediately** - Don't keep them alive unnecessarily
4. **Use try/finally** - Ensures cleanup even on errors
5. **Monitor memory usage** - Check for warnings and leaks
6. **Process in batches** - For very large operations
