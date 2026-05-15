---
story: 23
recommended_order: 23
phase: 4
title: Add rotation sync strategies and expanded CLI experimentation surface
labels:
- epic:rotation-strategies
- cli
- benchmark
- research
- testing
---

## Goal

Add explicit rotation-sync strategies and scheduling modes to the CLI and training system so HexNet rotation behavior becomes a controllable experimental surface rather than an implementation detail.

## Why

The project increasingly depends on questions like:

```text
Should rotations train independently?
When should rotations synchronize?
Should synchronization happen through weights or deltas?
Should all rotations participate equally?
Does schedule ordering matter?
```

These should be represented explicitly in:

- CLI arguments
- run manifests
- benchmark metadata
- tests
- future GPU implementations

This story intentionally focuses on scheduling/synchronization semantics before GPU acceleration.

## Scope

### New CLI arguments

- [ ] Add `--rotation-schedule`.
- [ ] Add `--sync-mode`.
- [ ] Add `--sync-interval`.
- [ ] Add `--sync-alpha` for EMA/blended strategies.
- [ ] Add `--rotation-batch-size` for future batching experiments.
- [ ] Add `--rotation-selection-mode` if needed for stochastic or sampled schedules.

### Initial scheduling modes

Support at least:

| Mode | Meaning |
|---|---|
| `sequential` | one rotation trains at a time according to `ro` |
| `independent` | rotations train independently before sync |
| `batched` | multiple rotations are processed together |

### Initial sync modes

Support at least:

| Mode | Meaning |
|---|---|
| `none` | rotations never synchronize |
| `weight_average` | average shared weights after interval |
| `delta_average` | combine updates before applying |
| `anchor_r0` | copy rotation 0 into equivalent slots |
| `ema` | exponential moving-average synchronization |

### Shared-edge bookkeeping

- [ ] Introduce a canonical edge-mapping abstraction.
- [ ] Define which rotation/block weights correspond to one another.
- [ ] Ensure sync logic does not depend on visualization structures.

### Benchmark support

- [ ] Add benchmark metadata fields describing schedule and sync strategy.
- [ ] Add benchmark-family documentation covering:
  - sequential vs independent rotation training
  - synchronization frequency sweeps
  - weight-average vs delta-average behavior
  - opposite-direction coupling experiments
  - train-subset-then-sync experiments
  - rotation specialization experiments

### Metrics

- [ ] Persist sync strategy metadata to runs.
- [ ] Record effective schedule history when feasible.
- [ ] Add benchmark metrics comparing:
  - convergence speed
  - directional transfer
  - interference
  - divergence between rotations before sync
  - stability after sync

## Suggested implementation structure

```text
src/networks/rotation_training/
  schedule.py
  sync.py
  canonical_edges.py
  planners.py
```

## Tests

- [ ] Tests proving equivalent slots synchronize correctly.
- [ ] Tests confirming `none` leaves rotations independent.
- [ ] Tests confirming `anchor_r0` copies correctly.
- [ ] Tests comparing weight-average vs delta-average semantics.
- [ ] Tests proving run metadata captures sync settings.
- [ ] Tests ensuring deterministic behavior under fixed seed.

## Acceptance criteria

- [ ] Rotation sync behavior is configurable entirely from the CLI.
- [ ] Runs persist schedule and sync metadata.
- [ ] At least two synchronization strategies are implemented and tested.
- [ ] Rotation scheduling/syncing no longer lives as ad-hoc logic inside one training loop.
- [ ] Benchmark documentation includes rotation-strategy comparison plans.
