---
story: 19
recommended_order: 19
phase: 3
title: Add no-lesion vs uniform vs directional retention benchmark
labels:
- epic:transfer-lesioning
- research
- paper
sync: {}
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
