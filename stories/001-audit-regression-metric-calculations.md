---
story: 1
recommended_order: 1
phase: 1
issue: 8
title: Audit and Fix Regression Metric Calculations
labels:
- tests
- research
- paper
- epic:core-correctness
sync:
  last_remote_updated: '2026-04-17T17:18:02Z'
  content_sha256: 55f7fa479323c2e38484c8f886b97bbfa66eec778f288950e8a723bd6b3d8ff3
---






## Checklist

- [ ] Review src/networks/metrics.py and document the intended meaning of each metric
- [ ] Verify accuracy proxy behavior for regression and decide whether to rename it in code and UI
- [ ] Verify r_squared and adjusted_r_squared formulas against known toy data
- [ ] Add unit tests for perfect fit, poor fit, and low-sample edge cases
- [ ] Add a short metrics interpretation section to docs
- [ ] Update any labels in Streamlit that are misleading for regression
- [ ] Streamlit - Add a metrics explainer expander on page showing formulas, caveats, and example values.

## Done When

- [ ] Metrics test pass
- [ ] Docs reflect final definitions
- [ ] streamlit labels match the final terminology
