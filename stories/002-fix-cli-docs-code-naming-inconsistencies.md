---
story: 2
recommended_order: 2
phase: 1
issue: 9
title: Fix CLI / docs /code naming inconsistencies
labels:
- documentation
- epic:core-correctness
sync:
  last_remote_updated: '2026-04-17T17:19:15Z'
  content_sha256: 35daa5380c4180c09892f52918c7e506c93dc7e6411466e337656e83bff865d4
---






## Why
There are a few naming drifts that will create friction later, such as the CLI dataset type linear while the dataset class registers as linear_scale, and stats_conmand.py having a typo in the filename.

## Checklist

- [ ] Normalize dataset naming between CLI options, dataset display names, docs, and manifests
- [ ] Rename stats_conmand.py to stats_command.py
- [ ] Confirm import paths remain stable after rename
- [ ] Update any docs that reference old names
- [ ] Add one smoke test covering the renamed stats command

## Streamlit

- [ ] Update dataset names in dropdowns and captions so they match CLI/docs exactly

## Definition of done

No 

- [ ] typoed command module names remain
- [ ] Dataset names are consistent across CLI, docs, saved configs, and Streamlit
