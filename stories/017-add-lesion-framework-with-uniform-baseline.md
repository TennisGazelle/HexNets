---
story: 17
recommended_order: 17
phase: 3
title: Add lesion framework with uniform baseline
labels:
- epic:transfer-lesioning
- research
- paper
- cli
sync: {}
---

## Goal

Introduce lesion abstractions with a uniform random edge-deletion baseline.

## Why

This is the path to your best claim: “geometry matters, not just sparsity.”

## Checklist

- [ ] Add a lesion abstraction layer
- [ ] Implement uniform random edge deletion baseline
- [ ] Support configuring lesion fraction and seed
- [ ] Persist lesion metadata to run config/manifest
- [ ] Add tests confirming lesions modify only intended weights

## Streamlit surface

- [ ] Add a Lesion Lab page with a lesion-type selector and lesion preview summary

## Acceptance criteria

- [ ] Uniform lesioning can be run, saved, compared, and reproduced
