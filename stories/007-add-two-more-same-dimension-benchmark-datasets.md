---
story: 6
recommended_order: 7
phase: 1
issue: 13
title: Add two more same-dimension benchmark datasets
labels:
- research
- paper
- epic:dataset-benchmarks
sync:
  last_remote_updated: '2026-04-14T16:37:57Z'
  content_sha256: 955d847f34e3fad77c1dcb42d2a85bbba276eb79659ddcee82af763b26f5a8a0
---


## Why
A first paper will be stronger if it is not basically “identity + linear scaling only.”

## Checklist

- [ ] Add one affine or permutation-based same-dimension dataset
- [ ] Add one nonlinear but controlled same-dimension dataset
- [ ] Add tests for output shapes and expected transforms
- [ ] Document why each dataset is useful as a benchmark
- [ ] Add sample generation examples to docs

## Streamlit

- [ ] Extend Dataset Explorer to preview all available datasets side by side

## Definition of done

- [ ] Benchmark suite includes at least 5 named datasets
- [ ] Each dataset is documented and selectable from CLI/Streamlit
