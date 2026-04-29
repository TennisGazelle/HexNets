---
story: 15
recommended_order: 15
phase: 3
title: Add sequential-task training workflow
labels:
- epic:transfer-lesioning
- research
- cli
sync:
  last_remote_updated: '2026-04-29T09:08:13Z'
  content_sha256: 211eaf515e238d12cd48c29912149838ada007e63e1bf702577f5dc6c5fd40bd
issue: 34
---


## Goal

Train Task A then Task B in one formal workflow with retention metrics.

## Why

Your living doc’s strongest near-term research path is sequential training and retention.

## Checklist

- [ ] Add a workflow for training Task A then Task B in one run
- [ ] Persist pre-task-B and post-task-B evaluation metrics
- [ ] Support selecting dataset A and dataset B independently
- [ ] Save retention metrics into run outputs
- [ ] Add tests for sequential-run artifact structure

## Streamlit surface

- [ ] Add a Sequential Task Lab page showing before/after metrics and forgetting curves

## Acceptance criteria

- [ ] You can measure retention degradation across two tasks using a single formal workflow
