---
story: 10
recommended_order: 5
phase: 1
title: Upgrade Streamlit from visualizer to experiment workbench
labels:
- epic:streamlit
- streamlit
- paper
sync:
  last_remote_updated: '2026-04-14T16:37:53Z'
  content_sha256: 477e4460ba1ea7e39c042278c47904ba01ab2220146c3015e0732e25ad8b206d
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
- [ ] Refactor shared UI helpers out of `src/hexnets_web/` (was monolithic `streamlit_app.py`)
- [ ] Document page layout in `docs/hexnets_web.md`

## Streamlit surface

Multi-page app shell plus the pages listed above (placeholders acceptable until those stories land).

## Acceptance criteria

- [ ] App has a clear research-workbench structure rather than a single-file demo feel
