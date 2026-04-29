---
story: 9
recommended_order: 10
phase: 2
title: Add aggregate run table and comparison utilities
labels:
- epic:experiment-orchestration
- paper
- streamlit
sync:
  last_remote_updated: '2026-04-29T09:08:01Z'
  content_sha256: 0accdd5828d161b10b13e16f48461a38df03a0fc12a94475abdb60c6ca99c2b3
issue: 20
---


## Goal

Scan persisted runs and compare or rank them without opening each folder by hand.

## Why

You have run folders, but not yet a strong compare-and-rank layer.

## Checklist

- [ ] Create a utility that scans runs/ and builds a consolidated dataframe/table
- [ ] Include final and best metrics per run
- [ ] Include filter columns for dataset, activation, loss, model, n, rotation, seed
- [ ] Export markdown and CSV summaries
- [ ] Add tests for parsing runs with missing or older fields

## Streamlit surface

- [ ] Add a Run Browser page with filters and sortable tables
- [ ] Let a user click into a run to see metadata + metrics + plots

## Acceptance criteria

- [ ] Completed runs can be compared without manually opening JSON files
