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
  last_remote_updated: '2026-04-29T09:07:40Z'
  content_sha256: 8189e5a71eb2156085dc27ec0bc7ba7b5001855fd68b69f1bb77bb9ffcf1b606
---







## Checklist

- [x] Review src/networks/metrics.py and document the intended meaning of each metric
- [x] Verify accuracy proxy behavior for regression and decide whether to rename it in code and UI
- [x] Verify r_squared and adjusted_r_squared formulas against known toy data
- [x] Add unit tests for perfect fit, poor fit, and low-sample edge cases
- [x] Add a short metrics interpretation section to docs
- [x] Update any labels in Streamlit that are misleading for regression
- [x] Streamlit - Add a metrics explainer expander on page showing formulas, caveats, and example values.

## Done When

- [x] Metrics test pass
- [x] Docs reflect final definitions
- [x] streamlit labels match the final terminology
