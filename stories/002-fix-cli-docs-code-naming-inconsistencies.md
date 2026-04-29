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
  last_remote_updated: '2026-04-29T09:07:42Z'
  content_sha256: 3067d0893908f4c2d1d17cf19c0c70f839996d14bd30d4f919f835654e0d6045
---








## Why
Resolved naming drift: CLI/manifest dataset id is now `linear_scale` (matches `display_name`); stats module is `stats_command.py`.

## Checklist

- [x] Normalize dataset naming between CLI options, dataset display names, docs, and manifests
- [x] Rename stats_conmand.py to stats_command.py
- [x] Confirm import paths remain stable after rename
- [x] Update any docs that reference old names
- [x] Add one smoke test covering the renamed stats command (`tests/test_commands_stats.py`)

## Streamlit

- [ ] Update dataset names in dropdowns and captions so they match CLI/docs exactly

## Definition of done

- [x] No typoed command module names remain
- [x] Dataset names are consistent across CLI, docs, saved configs, and Streamlit (explorer still identity-only; glossary documents `linear_scale`)
