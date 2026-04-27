---
story: 5
recommended_order: 6
phase: 1
issue: 12
title: Add Noisy Synthetic Regression Datasets
labels:
- research
- epic:dataset-benchmarks
sync:
  last_remote_updated: '2026-04-14T16:37:55Z'
  content_sha256: aed53f37cf63abc7868c29fbda601337d7d2b902590735a7b7e64434d899f35a
---


## Why
Noisy/fuzzy regression is a valuable proving ground

## Checklist

- [ ] Add configurable Gaussian noise to inputs
- [ ] Add configurable Gaussian noise to targets
- [ ] Support combined input+target noise
- [ ] Save noise parameters into run config
- [ ]  Add unit tests confirming noise application and shape stability
- [ ] Update docs with mathematical form and examples

## Streamlit

- [ ] Add a Dataset Explorer page showing clean vs noisy data samples
- [ ] Add sliders for noise level and a small preview chart/table

## Definition of done

- [ ] Noise can be turned on from CLI and Streamlit
- [ ] Run metadata makes noisy runs distinguishable from clean runs
