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
sync:
  last_remote_updated: '2026-04-29T09:08:17Z'
  content_sha256: cfd5a6364f2190f93aa63d7912bbbc30bbd1a9b93365fef1eac9defadb5686b6
issue: 36
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
