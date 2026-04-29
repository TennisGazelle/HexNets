---
story: 13
recommended_order: 13
phase: 2
title: Add a Benchmark Comparison page
labels:
- epic:streamlit
- streamlit
- paper
sync:
  last_remote_updated: '2026-04-29T09:08:09Z'
  content_sha256: eac39b5f9ebb03b73fc46ba902cb03aa4483bce7bb22124b719d446f5fc9bca9
issue: 32
---


## Goal

Compare multiple runs and models (HexNet vs MLP) in one place for paper drafting.

## Why

You need side-by-side metrics and curves instead of manual plot splicing.

## Checklist

- [ ] Select multiple runs for comparison
- [ ] Overlay training curves
- [ ] Compare final metrics in one table
- [ ] Allow HexNet vs MLP comparison
- [ ] Export a comparison table for paper drafting

## Streamlit surface

Benchmark Comparison page.

## Acceptance criteria

- [ ] You can compare candidate paper figures from the UI instead of manually splicing plots
