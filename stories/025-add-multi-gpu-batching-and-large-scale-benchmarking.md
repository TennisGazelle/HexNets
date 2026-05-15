---
story: 25
recommended_order: 25
phase: 6
title: Add multi-GPU batching infrastructure and large-scale benchmark orchestration
labels:
- epic:multi-gpu
- benchmarking
- infrastructure
- gpu
- research
---

## Goal

Scale HexNet benchmarking and rotation-training experiments across:

- multiple GPUs
- larger HexNet sizes
- larger benchmark matrices
- many rotation/sync strategies
- distributed benchmarking sweeps

This story targets environments such as Rosette-class systems with many CPU cores and multiple A100 GPUs.

## Why

Once the PyTorch backend exists, the project should evolve from:

```text
single experiment execution
```

toward:

```text
benchmark orchestration and large-scale research sweeps
```

The project is especially well-suited for batching because:

- rotations have identical tensor shapes
- many benchmark sweeps are embarrassingly parallel
- synchronization windows create natural batching boundaries
- benchmark families are highly combinatorial

## Research targets

Support systematic sweeps across:

- `n`
- dataset family
- activation function
- loss function
- rotation ordering
- epochs-per-rotation
- synchronization mode
- synchronization interval
- lesioning strategy
- nested-core strategy
- train/test directional splits

## Scope

### Multi-GPU support

- [ ] Add multi-GPU execution support.
- [ ] Evaluate:
  - `DistributedDataParallel`
  - process-per-GPU strategies
  - benchmark-level parallelization
  - rotation-level parallelization
- [ ] Support selecting GPU devices from CLI.
- [ ] Add infrastructure for large queued benchmark runs.

### Rotation batching

- [ ] Support batched execution across rotations.
- [ ] Evaluate tensor layouts optimized for batched rotation training.
- [ ] Benchmark:
  - sequential rotation training
  - independent rotation training
  - fully batched rotation execution

### Benchmark orchestration

- [ ] Add benchmark sweep configuration files.
- [ ] Add benchmark manifests summarizing sweep dimensions.
- [ ] Add resumable benchmark execution.
- [ ] Add failure recovery for partial sweep runs.
- [ ] Add aggregation scripts for benchmark results.

### Metrics and reporting

- [ ] Track:
  - throughput
  - GPU utilization
  - synchronization overhead
  - memory usage
  - scaling efficiency
  - convergence behavior
- [ ] Add summary-generation utilities.
- [ ] Add comparison tooling across benchmark families.

### Infrastructure

- [ ] Add benchmark queueing/orchestration utilities.
- [ ] Add support for headless cluster execution.
- [ ] Add structured run metadata suitable for large experiment archives.
- [ ] Add optional artifact pruning/compression strategy.

## Suggested implementation structure

```text
src/benchmarking/
  sweeps/
  orchestration/
  aggregation/
  distributed/

configs/benchmark_sweeps/
```

## Tests

- [ ] Multi-GPU smoke test.
- [ ] Rotation-batching correctness tests.
- [ ] Distributed determinism test where feasible.
- [ ] Sweep resumption tests.
- [ ] Aggregation integrity tests.

## Benchmark targets

- [ ] Demonstrate training beyond current practical NumPy limits.
- [ ] Benchmark scaling across increasing `n`.
- [ ] Benchmark scaling across multiple GPUs.
- [ ] Compare batching strategies.
- [ ] Compare synchronization overheads.

## Acceptance criteria

- [ ] Benchmark sweeps can execute across multiple GPUs.
- [ ] Rotation batching works for at least one benchmark family.
- [ ] Benchmark orchestration supports resumable large-scale runs.
- [ ] Aggregate reporting exists for sweep comparisons.
- [ ] The system is capable of exploiting Rosette-class multi-GPU infrastructure.
