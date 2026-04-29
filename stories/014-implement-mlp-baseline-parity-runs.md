---
story: 14
recommended_order: 14
phase: 3
title: Implement MLP baseline parity runs
labels:
- epic:transfer-lesioning
- research
- paper
- cli
sync:
  last_remote_updated: '2026-04-29T09:08:11Z'
  content_sha256: 24e208c465b25a57b320111a55e3909c77022213087e5fb6ac9330ca459abded
issue: 33
---


## Goal

Make MLP a first-class baseline alongside HexNet in sweeps and comparisons.

## Why

The repo already has MLPNetwork, but it needs to become a proper baseline in the experiment workflow, not just a side model.

## Checklist

- [ ] Make MLP baseline selectable in sweep command
- [ ] Record model parameter count for HexNet and MLP
- [ ] Document how hidden dims are chosen
- [ ] Add comparison tests ensuring MLP and HexNet runs serialize in a common format
- [ ] Add example benchmark doc section for HexNet vs MLP

## Streamlit surface

- [ ] Add MLP vs HexNet toggle to Benchmark Comparison page

## Acceptance criteria

- [ ] HexNet claims can always be contextualized against an MLP baseline
