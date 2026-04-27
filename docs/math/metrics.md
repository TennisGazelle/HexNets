Assumptions in snippets:

* `y_target` = true target vector for one training example (shape: `(n,)`)
* `y_pred` = model prediction vector for one training example (shape: `(n,)`)
* `n` = output dimensions per example
* `count` = number of examples accumulated in the current epoch
* Metrics are accumulated via `Metrics.tally_accurcy_r2(...)` and finalized via `Metrics.calc_accuracy_r2(...)`

---

# Regression Metrics Glossary (`src/networks/metrics.py`)

This glossary documents the intended meaning of each metric/state field currently used during training.

**Plot / UI caveat:** training figures may set `accuracy_detail = "RMSE"` while the stored `accuracy` series is **`exp(-RMSE)`** (bounded score, not RMSE). Treat axis subtitles as legacy labels unless code is updated to match.

## 1) `loss`

**meaning:** epoch-level optimization objective reported by the configured loss function (for example MSE, Huber, etc.).

**in code:** appended via `add_metric(loss, accuracy, r_squared, adjusted_r_squared)`.

---

## 2) `accuracy` (regression proxy score)

**meaning:** a bounded "goodness" score for regression, not classification accuracy.

**per-example definition:**

* compute `rmse = sqrt(mean((y_pred - y_target)^2))`
* compute `score = exp(-rmse)`

**epoch definition:**

* `accuracy = mean(score over examples in epoch) = correct / count`

**range and interpretation:**

* in `(0, 1]`
* `1.0` means perfect prediction (`rmse = 0`)
* decreases smoothly toward `0` as prediction error grows

---

## 3) `r_squared`

**meaning:** coefficient of determination over all accumulated scalar targets in the epoch.

**accumulated state:**

* `ss_res_sum = sum((y_pred - y_target)^2)` over all examples and dimensions
* `y_sum = sum(y_target)` over all examples and dimensions
* `y2_sum = sum(y_target^2)` over all examples and dimensions
* `num_elems = count * n`

**computed totals:**

* `ss_tot = y2_sum - (y_sum^2 / num_elems)`
* `r_squared = 1 - ss_res_sum / (ss_tot + 1e-12)`

**interpretation:**

* `1.0` is perfect fit
* `0.0` is "predicting the mean"-level fit
* `< 0.0` is worse than mean prediction baseline

---

## 4) `adjusted_r_squared`

**meaning:** `r_squared` corrected for model complexity.

**formula used:**

* `adjusted_r_squared = 1 - (1 - r_squared) * (num_elems - 1) / (num_elems - p - 1)`
* where `p` is the chosen feature/parameter count (defaults to `n` in `Metrics.calc_accuracy_r2`)

**edge-case behavior in code:**

* if `num_elems <= p + 1`, it falls back to `adjusted_r_squared = r_squared`

---

## 5) Internal accumulator fields

**`correct`**
* running sum of per-example regression proxy scores (`exp(-rmse)`), later averaged into `accuracy`.

**`ss_res_sum`**
* running residual sum of squares numerator for `r_squared`.

**`y_sum`**
* running sum of all target values used to compute total variance.

**`y2_sum`**
* running sum of squared target values used to compute total variance.

**`count`**
* number of examples accumulated for the current epoch.

---

## Notes and caveats

* The field named `accuracy` is intentionally a regression proxy score; consider labeling it as "RMSE score" in user-facing plots/tables to avoid classification ambiguity.
* `r_squared` and `adjusted_r_squared` are epoch-level aggregate statistics, not per-example values.
* The method name `tally_accurcy_r2` contains a spelling typo but is functionally the accumulator used by training loops.
