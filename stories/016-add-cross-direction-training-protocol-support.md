---
story: 16
recommended_order: 16
phase: 3
title: Add cross-direction training protocol support
labels:
- epic:transfer-lesioning
- research
- cli
sync:
  last_remote_updated: '2026-04-29T09:08:15Z'
  content_sha256: aae758b5a1ab09fffc521c11ef2f56154f26440fb2e90f92c76448bb76d48e23
issue: 35
---


## Goal

Support multiple directional training protocols as first-class config, not one-off scripts.

## Why

Directional transfer experiments need explicit, reproducible protocols.

## Checklist

- [ ] Implement protocol A: train one direction then another
- [ ] Implement protocol B: alternate directions every k epochs
- [ ] Persist the chosen protocol in run config
- [ ] Add tests for protocol selection and metadata
- [ ] Add docs describing how each protocol maps to your research question

## Streamlit surface

- [ ] Add protocol selector and explanation to Sequential Task Lab

## Acceptance criteria

- [ ] Directional transfer experiments are a first-class feature instead of custom scripts
