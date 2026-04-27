---
story: 20
recommended_order: 20
phase: 4
title: Add paper-figure export and traceability mapping
labels:
- epic:paper-support
- paper
- docs
sync: {}
---

## Goal

Export figures with a manifest tying each file to run IDs and configs for reproducibility.

## Why

Paper figures must be reproducible from the codebase and run artifacts.

## Checklist

- [ ] Add a script or CLI command that exports selected run plots into a paper-figures directory
- [ ] Generate a manifest mapping figure file → run ID → config
- [ ] Add markdown snippets for figure captions / notes
- [ ] Document how to reproduce each exported figure

## Streamlit surface

- [ ] Add an “Export for paper” action from Run Browser / Benchmark Comparison page

## Acceptance criteria

- [ ] Every figure you use in the paper has a reproducible provenance trail
