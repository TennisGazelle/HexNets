---
story: 18
recommended_order: 18
phase: 3
title: Add directional lesion types
labels:
- epic:transfer-lesioning
- research
- paper
- streamlit
sync:
  last_remote_updated: '2026-04-29T09:08:19Z'
  content_sha256: 09b2c0fb0efaefdcd62d7b8b3ae3ac1ec7bd2eff02bc065d21e2a69b5a20b903
issue: 37
---


## Goal

Extend lesions beyond uniform deletion with direction-aware lesion modes.

## Why

Directional structure is central to the HexNet research narrative.

## Checklist

- [ ] Implement single-direction lesion
- [ ] Implement opposing-direction lesion
- [ ] Implement sector lesion
- [ ] Implement rotating-direction lesion
- [ ] Add validation and metadata for all lesion modes
- [ ] Add tests confirming targeted scope of each lesion type

## Streamlit surface

- [ ] In Lesion Lab, visualize which structural region/direction is being lesioned
- [ ] Show side-by-side lesion mask summaries

## Acceptance criteria

- [ ] Directional lesions are operational and inspectable
