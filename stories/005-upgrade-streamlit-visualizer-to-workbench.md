---
story: 10
recommended_order: 5
phase: 1
title: Upgrade Streamlit from visualizer to experiment workbench
labels:
- paper
- epic:streamlit
- streamlit
sync:
  last_remote_updated: '2026-04-29T09:07:50Z'
  content_sha256: 149619396a9ce85857e133a85ced209d8761fb1f31638b496c1a2d7980bb3230
issue: 14
---



## Goal

Restructure the Streamlit app into a multi-page research workbench instead of a narrow demo.

## Why

The current app is useful, but still narrow: two tabs, one quick training button, and reference image loading.

## Checklist

- [ ] Split app into multi-page navigation
- [ ] Keep current Network Explorer page
- [ ] Keep current Rotation Comparison page
- [ ] Add placeholders/routes for Dataset Explorer, Run Browser, Sweep Planner, Lesion Lab
- [ ] Refactor shared UI helpers out of streamlit_app.py
- [ ] Document page layout in docs/streamlit_app.md

## Streamlit surface

Multi-page app shell plus the pages listed above (placeholders acceptable until those stories land).

## Acceptance criteria

- [ ] App has a clear research-workbench structure rather than a single-file demo feel
