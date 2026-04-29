---
story: 3
recommended_order: 3
phase: 1
issue: 10
title: Add Run Manifest Fields needed for paper traceability
labels:
- paper
- epic:core-correctness
sync:
  last_remote_updated: '2026-04-29T09:07:45Z'
  content_sha256: d7367c034f05f465cee1fdf75efb9888b104c70fb95764764eb68c79ec4bf551
---






## Why
Run persistence exists already, but it is still fairly minimal for paper traceability.

## Checklist

- [ ] Add git commit SHA to run manifest
- [ ] Add random seed to persisted config/manifest
- [ ] Add dataset metadata block to persisted config
- [ ] Add model parameter count to manifest
- [ ] Add optional freeform note / tag field for experiments
- [ ] Preserve backward compatibility when loading older runs (not important, but it'd be nice,  we can delete older runs if it helps)

## Streamlit

- [ ]  Add a Run Metadata panel for any loaded run

## Definition of done

- [ ] New runs contain paper-useful provenance
- [ ] Old runs still load cleanly
