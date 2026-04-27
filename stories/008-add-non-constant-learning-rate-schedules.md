---
story: 8
recommended_order: 8
phase: 2
title: Add non-constant learning-rate schedules
labels:
- epic:experiment-orchestration
- cli
- tests
- streamlit
sync: {}
---

## Goal

Implement real learning-rate schedules beyond the constant plugin stub.

## Why

The architecture for learning-rate plugins exists, but only constant is actually implemented.

## Checklist

- [ ] Implement exponential decay schedule
- [ ] Implement one additional schedule, likely rolling decay or step decay
- [ ] Add tests for rate_at_iteration() behavior
- [ ] Ensure schedules serialize cleanly into run config
- [ ] Update reference generation so learning-rate figures include new schedules

## Streamlit surface

- [ ] Add a Learning Rate Schedules page using the existing reference-figure concept
- [ ] Add interactive plots comparing schedules

## Acceptance criteria

- [ ] At least 3 schedules are selectable from CLI and viewable in Streamlit
