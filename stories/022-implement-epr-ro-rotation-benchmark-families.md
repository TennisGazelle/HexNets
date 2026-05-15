---
story: 22
recommended_order: 22
phase: 4
title: Implement EPR/RO rotation scheduling and benchmark-family coverage
labels:
- epic:rotation-strategies
- benchmark
- cli
- documentation
- research
---

## Goal

Fully implement `--epr` / `--epochs-per-rotation` and `--ro` / `--rotation-ordering` as real rotation scheduling controls, then expand the benchmark-family documentation so major HexNet research tracks are represented as runnable or planned benchmark families.

## Why

`epr` and `ro` currently exist as CLI-facing concepts but are not yet complete scheduling primitives. Before introducing GPU backends or more aggressive rotation-sync strategies, the project needs a stable CPU/reference interpretation of:

```text
which rotations train
how long each rotation trains
when rotation changes happen
how those choices are recorded in benchmark metadata
```

This story also prevents the benchmark surface from narrowing too much around ordinary dataset sweeps. Previous project notes include rotational attractors, directional transfer, nested/hex-core swapping, lesioning, and ant-maze ideas. These should be visible in markdown as benchmark families or research tracks even when implementation is deferred.

## Scope

### CLI behavior

- [ ] Make `--epr` control how many epochs are trained before the active rotation changes.
- [ ] Make `--ro` define the explicit rotation order used when `--epr` is set.
- [ ] Define default `ro` behavior when `epr` is set and `ro` is omitted.
- [ ] Define behavior when `ro` is supplied without `epr`.
- [ ] Persist `epr` and `ro` into run config and manifests.
- [ ] Ensure resumed runs preserve or explicitly override rotation schedule metadata.

### Scheduling behavior

- [ ] Add a small rotation schedule abstraction rather than embedding the logic directly into the train loop.
- [ ] Support at least sequential rotation scheduling:

```text
train r0 for epr epochs
train r1 for epr epochs
train r2 for epr epochs
...
repeat until total epochs are reached
```

- [ ] Support repeated or partial rotation orderings, such as:

```text
--ro 0 1 2
--ro 0 0 3
--ro 0 2 4 1 3 5
```

- [ ] Validate all rotations are in `0..5`.
- [ ] Validate `epr` is compatible with total epochs.
- [ ] Define how partial final EPR windows behave.

### Benchmark-family documentation

- [ ] Add or update benchmark-family markdown so the following are represented:
  - [ ] ordinary MLP-vs-HexNet dataset sweeps
  - [ ] rotation-ordering sweeps
  - [ ] epochs-per-rotation sweeps
  - [ ] train-one-rotation/test-all-rotations transfer
  - [ ] train-subset-rotations/test-held-out-rotations transfer
  - [ ] alternating rotation interference
  - [ ] rotational attractor / latent orbit dynamics
  - [ ] attractor splitting / persona-like regime experiments
  - [ ] directional lesioning and uniform lesion baselines
  - [ ] nested-core initialization
  - [ ] hex-core swapping / transplant experiments
  - [ ] frozen vs trainable inner-core experiments
  - [ ] ant-maze / queryable hex lattice experiments as deferred system benchmarks

### Traceability

- [ ] Link benchmark-family markdown to existing vision docs where applicable.
- [ ] Ensure every speculative track has either:
  - a benchmark-family markdown stub, or
  - a clear note explaining why it is not yet a benchmark family.

## Suggested files

```text
src/networks/rotation_schedule.py
src/services/benchmark_families/
docs/benchmarks/rotation-strategies.md
docs/benchmarks/directional-transfer.md
docs/benchmarks/rotational-attractors.md
docs/benchmarks/nested-cores.md
docs/benchmarks/ant-maze.md
```

The exact paths can change if the current documentation structure suggests a better home.

## Tests

- [ ] Unit tests for schedule expansion from `(epochs, epr, ro)`.
- [ ] CLI validation tests for invalid rotations.
- [ ] CLI validation tests for `ro` without `epr`.
- [ ] Run-config tests confirming `epr` and `ro` are persisted.
- [ ] Regression test proving existing non-rotating training behavior remains unchanged when `epr` is omitted.

## Acceptance criteria

- [ ] `hexnet train --epr ... --ro ...` actually changes rotation during training.
- [ ] Runs record the effective rotation schedule.
- [ ] At least one benchmark-family markdown file explicitly covers rotation scheduling experiments.
- [ ] Rotational attractors, directional transfer, nested-core/hex-core swapping, lesioning, and ant-maze concepts are represented in markdown.
- [ ] Existing default training behavior remains stable when no rotation schedule flags are provided.
