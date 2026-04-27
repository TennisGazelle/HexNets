Assumptions in snippets:

* `y_target` = true target vector for one training example (shape: `(n,)`)
* `y_pred` = model prediction vector for one training example (shape: `(n,)`)
* `n` = output dimensions per example
* `count` = number of examples accumulated in the current epoch
* Metrics are accumulated via `Metrics.tally_regression_score_r2(...)` and finalized via `Metrics.calc_regression_score_and_r2(...)`

---

# Regression metrics glossary (`src/networks/metrics.py`)

## 1) `loss`

**Meaning:** epoch-level optimization objective from the configured loss (MSE, Huber, etc.).

**In code:** appended via `add_metric(loss, regression_score, r_squared, adjusted_r_squared)`.

---

## 2) `regression_score`

**Meaning:** mean over training examples of `exp(-RMSE)` per example (bounded regression score). **Not** classification accuracy.

**Per example:**

* `rmse = sqrt(mean((y_pred - y_target)^2))` over output dimensions
* `score = exp(-rmse)`

**Epoch value:**

* `regression_score = mean(score) = regression_score_sum / count`

**Range:** `(0, 1]`; `1.0` is zero error.

---

## 3) `r_squared`

**Meaning:** coefficient of determination over all scalar targets accumulated in the epoch.

**State:** `ss_res_sum`, `y_sum`, `y2_sum`, `count`, and `n` as in code.

**Formula:** `ss_tot = y2_sum - (y_sum^2 / (count * n))`, then `r_squared = 1 - ss_res_sum / (ss_tot + 1e-12)`.

---

## 4) `adjusted_r_squared`

**Meaning:** `r_squared` adjusted for complexity:
`adjusted_r_squared = 1 - (1 - r_squared) * (num_elems - 1) / (num_elems - p - 1)` with `num_elems = count * n`, `p` from the trainer (often `n` for hex nets). If `num_elems <= p + 1`, code uses `adjusted_r_squared = r_squared`.

---

## 5) Internal fields (`as_dict` / checkpointing)

* **`regression_score_sum`** — sum of per-example `exp(-rmse)`; divided by `count` for epoch `regression_score`.
* **`ss_res_sum`**, **`y_sum`**, **`y2_sum`**, **`count`** — as used for `r_squared`.

---

## Figures (`TrainingFigure`)

Second subplot title: **Regression score** (`regression_score_detail` subtitle). Y-axis label: **Mean exp(-RMSE)**.

---

## Notes

* `r_squared` / `adjusted_r_squared` are epoch aggregates, not per-example series.
* Legacy pickles/JSON using `accuracy` / `correct` are incompatible with this naming; retrain or migrate keys if needed.
