---
story: 11
recommended_order: 11
phase: 2
title: Add a Run Browser page
labels:
- epic:streamlit
- streamlit
- paper
sync:
  last_remote_updated: '2026-04-29T09:08:04Z'
  content_sha256: 1471f58b5deaaf4f92e112c522fba35dec93e08c5229a027f9f7ac9c3e1a3be5
issue: 30
---


## Goal

Inspect completed runs entirely from Streamlit.

## Why

Completed runs live under `runs/`; this story adds the dedicated Streamlit surface to browse and open them without using the CLI or raw JSON.

## Checklist

- [ ] List runs from runs/
- [ ] Filter by model, dataset, n, rotation, activation, loss
- [ ] Show run metadata and persisted config
- [ ] Show training curves for selected run
- [ ] Gracefully handle missing legacy fields

## Streamlit surface

Run Browser page (may overlap with Story 9 utilities; converge on one UX).

## Acceptance criteria

- [ ] A completed run can be inspected entirely from Streamlit
