---
story: 7
recommended_order: 9
phase: 2
title: Build a sweep command for batch experiments
labels:
- epic:experiment-orchestration
- cli
- research
- paper
sync:
  last_remote_updated: '2026-04-29T09:07:59Z'
  content_sha256: c8d9a1962f0dea7550c5dfd2fd2824ce44dea481b15435324bb1a53eceed8df9
issue: 19
---


## Goal

Add a first-class CLI for controlled parameter sweeps instead of ad hoc shell loops.

## Why

You already have single-run training. What is missing is a first-class mechanism for controlled sweeps.

## Checklist

- [ ] Add a new CLI command for parameter sweeps
- [ ] Support sweeping over dataset, activation, loss, model, n, rotation, and learning rate schedule
- [ ] Skip or resume already-completed runs
- [ ] Emit a summary CSV/JSON of completed sweep outputs
- [ ] Add tests for sweep plan generation
- [ ] Document usage examples

## Streamlit surface

- [ ] Add a Sweep Planner page that previews the run matrix before execution
- [ ] Show total planned runs and estimated artifact count

## Acceptance criteria

- [ ] You can define a repeatable sweep without shell scripting it by hand
- [ ] Sweep outputs aggregate into a summary artifact
