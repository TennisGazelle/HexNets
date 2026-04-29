---
story: 4
recommended_order: 4
phase: 1
issue: 11
title: Promote Dataset Registry to First-class CLI Support
labels:
- tests
- epic:dataset-benchmarks
sync:
  last_remote_updated: '2026-04-29T09:07:47Z'
  content_sha256: 8ff22e2ea91e329b3de8330ea1ab320be0e9ee2baad6ad4f7eb24b94c128b339
---






## Why
You already have more dataset structure than the CLI exposes. Right now the command helper hardcodes only identity and linear.

## Checklist

- [ ] Refactor dataset selection to use the dataset registry instead of hardcoded branching
- [ ] Expose diagonal_scale through CLI
- [ ] Ensure saved run config stores the exact dataset display name
- [ ] Add tests for selecting each registered dataset
- [ ] Document how to add new datasets

## Streamlit

- [ ] Replace the fixed training dataset behavior with a dataset selector using the registry

## Definition of done

- [ ] Adding a dataset no longer requires multiple manual switch statements
- [ ] CLI and Streamlit both discover the same dataset list
