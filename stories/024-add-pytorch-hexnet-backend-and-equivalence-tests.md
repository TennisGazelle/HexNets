---
story: 24
recommended_order: 24
phase: 5
title: Add PyTorch HexNet backend with equivalence testing and GPU acceleration
labels:
- epic:torch-backend
- gpu
- benchmarking
- testing
- research
---

## Goal

Implement a fast PyTorch-based HexNet backend that supports:

- GPU execution
- large `n`
- rotation-aware training
- explicit synchronization strategies
- debugging against the existing NumPy implementation

The NumPy implementation remains the reference/debugging implementation.

## Why

The current NumPy implementation is valuable for:

- visualization
- correctness
- inspectability
- experimentation

But it does not scale well for large HexNet sizes.

A dense global matrix representation becomes prohibitively expensive as `n` grows. The PyTorch backend should instead treat each rotation as an independent dense block-path network and synchronize equivalent weights explicitly.

This backend is intended to become the primary benchmarking/training engine.

## Core architecture

### Rotation-local block representation

Represent each rotation as a stack of dense adjacent-layer matrices.

Instead of:

```text
full global adjacency matrix
```

use:

```text
rotation r
  block 0: [n, n+1]
  block 1: [n+1, n+2]
  ...
```

### Recommended tensor shape

Prefer stacked rotation tensors:

```text
W[layer].shape = [rotation, in_dim, out_dim]
```

rather than deeply nested Python lists.

### Reference/debugging relationship

The NumPy backend remains:

- the reference implementation
- the visualization implementation
- the equivalence-testing target

The PyTorch backend should expose export/import helpers to compare behavior.

## Scope

### Backend implementation

- [ ] Add a PyTorch HexNet implementation.
- [ ] Support CPU and CUDA execution.
- [ ] Add device selection (`cpu`, `cuda`, `auto`).
- [ ] Implement forward pass using block-layer matrices.
- [ ] Implement rotation-local training.
- [ ] Implement sync hooks compatible with Story 023.
- [ ] Add save/load support.

### Debugging helpers

- [ ] Export PyTorch rotation weights to NumPy.
- [ ] Import NumPy rotation weights into PyTorch.
- [ ] Add deterministic seeding helpers.
- [ ] Add debugging utilities for comparing:
  - forward outputs
  - losses
  - gradients/deltas
  - synchronized weights

### CLI surface

- [ ] Add `--backend numpy|torch`.
- [ ] Add `--device cpu|cuda|auto`.
- [ ] Ensure existing CLI commands work with backend selection.

### Benchmarking

- [ ] Add benchmark scripts comparing NumPy vs PyTorch runtime.
- [ ] Add scaling benchmarks for larger `n`.
- [ ] Benchmark memory usage.
- [ ] Benchmark sync-strategy overhead.

### GPU support

- [ ] Confirm CUDA execution works.
- [ ] Confirm model parameters move to GPU correctly.
- [ ] Confirm training loop executes without CPU fallback.
- [ ] Add optional mixed precision experimentation if stable.

## Suggested implementation structure

```text
src/networks/
  HexagonalTorchNetwork.py

src/networks/hex_torch/
  model.py
  layers.py
  sync.py
  debug.py
  benchmarks.py
```

## Tests

### Equivalence tests

For small `n`:

- [ ] NumPy forward ~= PyTorch forward.
- [ ] NumPy loss ~= PyTorch loss.
- [ ] One-step updates remain numerically comparable.
- [ ] Sync logic preserves equivalent slots.

### Backend tests

- [ ] CUDA availability test.
- [ ] CPU/GPU parity test.
- [ ] Save/load parity test.
- [ ] Deterministic seed test.

### Performance tests

- [ ] Benchmark proving PyTorch backend is faster than NumPy for moderate/large `n`.
- [ ] Benchmark demonstrating GPU acceleration.

## Acceptance criteria

- [ ] PyTorch backend trains successfully on GPU.
- [ ] Existing CLI can select backend.
- [ ] Small-model equivalence tests pass against NumPy.
- [ ] Large `n` training is measurably faster than NumPy.
- [ ] Rotation scheduling and synchronization integrate with the backend cleanly.
