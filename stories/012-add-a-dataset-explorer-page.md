---
story: 12
recommended_order: 12
phase: 2
title: Add a Dataset Explorer page
labels:
- epic:streamlit
- streamlit
- research
sync:
  last_remote_updated: '2026-04-29T09:08:07Z'
  content_sha256: 190b30a3359cd65f9e0cae9f9b59b50c03bf4673866e788a03a7fa8e5a42751e
issue: 31
---


## Goal

Preview benchmark datasets (including noise) before training.

## Why

You should be able to preview how a dataset looks (and how noise affects it) before committing to long training runs.

## Checklist

- [ ] Select dataset type
- [ ] Adjust sample count and dimension
- [ ] Adjust noise settings where applicable
- [ ] Show a sample table and summary stats
- [ ] Add a short description of what transformation the dataset represents

## Streamlit surface

Dataset Explorer page.

## Acceptance criteria

- [ ] You can visually inspect benchmark datasets before training on them
