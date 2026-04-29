---
story: 19
recommended_order: 19
phase: 3
title: Add no-lesion vs uniform vs directional retention benchmark
labels:
- epic:transfer-lesioning
- research
- paper
sync:
  last_remote_updated: '2026-04-29T09:08:21Z'
  content_sha256: 548527f65815bb25cc6ff43f5ea473dd4819a9651ab73374f954936ae6b9df00
issue: 38
---


## Goal

One reproducible benchmark comparing retention under no lesion, uniform lesion, and directional lesion.

## Why

This is the most important medium-term story in the whole backlog (geometry vs sparsity).

## Checklist

- [ ] Run sequential-task workflow without lesion
- [ ] Run sequential-task workflow with uniform lesion
- [ ] Run sequential-task workflow with directional lesion
- [ ] Export one summary table comparing retention degradation
- [ ] Export one plot intended for paper use
- [ ] Add doc section describing interpretation boundaries

## Streamlit surface

- [ ] Add a one-click comparison view in Lesion Lab

## Acceptance criteria

- [ ] You have a reproducible benchmark that can support the geometry-vs-sparsity argument
