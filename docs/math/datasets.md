Assumptions in snippets:

* `self.num_samples` = N
* `self.d` = input/output dim
* `rng = np.random.default_rng(self.seed)`
* Inputs typically in `[-1, 1]`

---

# Group A — Linear & affine operators

## 1) `IdentityDataset`

**math:** ( y = x )
**description:** perfect passthrough.
**numpy implementation:**

```python
X = (rng.random((self.num_samples, self.d)) * 2 - 1).astype(float)
Y = X.copy()
```

**good for:** sanity checks; debugging shapes; baseline convergence.

---

## 2) `LinearScaleDataset`

**math:** ( y = s x )
**description:** global scaling.
**numpy implementation:**

```python
X = (rng.random((self.num_samples, self.d)) * 2 - 1).astype(float)
Y = X * self.scale  # e.g. 0.5, 2.0
```

**good for:** checking optimizer sensitivity to magnitude; learning-rate tuning.

---

## 3) `DiagonalLinearDataset`

**math:** ( y_i = a_i x_i ) (diagonal (A))
**description:** per-dimension scaling with varying conditioning.
**numpy implementation:**

```python
X = (rng.random((self.num_samples, self.d)) * 2 - 1).astype(float)
a = rng.uniform(self.min_scale, self.max_scale, size=(self.d,))
Y = X * a
```

**good for:** per-dimension learning behavior; Adam vs SGD comparisons.

---

## 4) `FullLinearDataset`

**math:** ( y = A x )
**description:** dense mixing; controllable difficulty via condition number.
**numpy implementation:**

```python
X = (rng.standard_normal((self.num_samples, self.d))).astype(float)
A = rng.standard_normal((self.d, self.d)).astype(float) * self.A_scale
Y = X @ A.T
```

**good for:** learning cross-feature coupling; stress-testing depth/width.

---

## 5) `AffineDataset`

**math:** ( y = A x + b )
**description:** adds bias/translation.
**numpy implementation:**

```python
X = (rng.standard_normal((self.num_samples, self.d))).astype(float)
A = rng.standard_normal((self.d, self.d)).astype(float) * self.A_scale
b = rng.uniform(-self.b_scale, self.b_scale, size=(self.d,)).astype(float)
Y = X @ A.T + b
```

**good for:** ensuring your model has bias terms and learns offsets.

---

## 6) `OrthogonalRotationDataset`

**math:** ( y = Q x ) where (Q^\top Q = I)
**description:** pure rotation/reflection; preserves norms.
**numpy implementation:**

```python
X = rng.standard_normal((self.num_samples, self.d)).astype(float)
M = rng.standard_normal((self.d, self.d))
Q, _ = np.linalg.qr(M)  # orthogonal
Y = X @ Q.T
```

**good for:** testing invariances; cosine similarity losses; stable transforms.

---

# Group B — Elementwise nonlinearities (still deterministic)

## 7) `ElementwisePowerDataset`

**math:** ( y = \operatorname{sign}(x),|x|^p )
**description:** smooth-ish nonlinearity with tunable curvature.
**numpy implementation:**

```python
X = (rng.random((self.num_samples, self.d)) * 2 - 1).astype(float)
Y = np.sign(X) * (np.abs(X) ** self.p)  # e.g. p=2,3
```

**good for:** nonlinearity learning; activation choice experiments.

---

## 8) `SineDataset`

**math:** ( y = \sin(\omega x) )
**description:** periodic mapping; frequency controls difficulty.
**numpy implementation:**

```python
X = (rng.random((self.num_samples, self.d)) * 2 - 1).astype(float)
Y = np.sin(self.omega * X)
```

**good for:** testing approximation capacity; under/overfitting behavior.

---

## 9) `SoftThresholdDataset`

**math:** ( y = \text{sign}(x)\max(|x|-\lambda, 0) )
**description:** shrinkage operator (prox of L1).
**numpy implementation:**

```python
X = (rng.standard_normal((self.num_samples, self.d))).astype(float)
Y = np.sign(X) * np.maximum(np.abs(X) - self.lam, 0.0)
```

**good for:** “optimization operator” learning; sparsity behavior; robust losses.

---

# Group C — Denoising & corruption (operator learning, robust)

## 10) `GaussianDenoiseDataset`

**math:** input ( \tilde{x} = x + \epsilon), target (y=x)
**description:** learn to remove Gaussian noise.
**numpy implementation:**

```python
Y = (rng.standard_normal((self.num_samples, self.d))).astype(float)
X = Y + rng.normal(0.0, self.sigma, size=Y.shape)
```

**good for:** robustness; Huber/LogCosh vs MSE comparisons.

---

## 11) `SaltPepperDenoiseDataset`

**math:** random coordinates replaced by ±1; target (y=x)
**description:** impulsive noise (outliers).
**numpy implementation:**

```python
Y = (rng.random((self.num_samples, self.d)) * 2 - 1).astype(float)
X = Y.copy()
mask = rng.random(X.shape) < self.p  # corruption prob
X[mask] = rng.choice([-1.0, 1.0], size=mask.sum())
```

**good for:** outlier-robust training; Huber / quantile.

---

## 12) `MaskDenoiseDataset`

**math:** random dimensions set to 0; target (y=x)
**description:** “inpainting” in vector form.
**numpy implementation:**

```python
Y = (rng.standard_normal((self.num_samples, self.d))).astype(float)
X = Y.copy()
mask = rng.random(X.shape) < self.p
X[mask] = 0.0
```

**good for:** completion; structured generalization; potentially add mask-channel later.

---

# Group D — Projection / constraint satisfaction

## 13) `UnitSphereProjectionDataset`

**math:** ( y = x / |x|_2 )
**description:** normalize vectors to unit norm.
**numpy implementation:**

```python
X = rng.standard_normal((self.num_samples, self.d)).astype(float)
norm = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
Y = X / norm
```

**good for:** direction learning; cosine loss; stability tests.

---

## 14) `L2BallProjectionDataset`

**math:** ( y = x ) if (|x|\le r) else ( y = r x/|x| )
**description:** clipping by L2 radius.
**numpy implementation:**

```python
X = rng.standard_normal((self.num_samples, self.d)).astype(float)
norm = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
scale = np.minimum(1.0, self.r / norm)
Y = X * scale
```

**good for:** bounded outputs; robust training; “prox-like” learning.

---

## 15) `NonNegativeProjectionDataset`

**math:** ( y = \max(x, 0) )
**description:** project onto nonnegative orthant.
**numpy implementation:**

```python
X = (rng.standard_normal((self.num_samples, self.d))).astype(float)
Y = np.maximum(X, 0.0)
```

**good for:** piecewise-linear learning; ReLU output experiments.

---

## 16) `SimplexProjectionDataset`

**math:** ( y = \Pi_\Delta(x) ) (projection onto simplex: (y\ge0, \sum y=1))
**description:** turns arbitrary vectors into probability vectors.
**numpy implementation (simple, not fastest):**

```python
X = rng.standard_normal((self.num_samples, self.d)).astype(float)

def proj_simplex(v):
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, v.size+1) > (cssv - 1))[0][-1]
    theta = (cssv[rho] - 1) / (rho + 1)
    return np.maximum(v - theta, 0)

Y = np.stack([proj_simplex(x) for x in X], axis=0)
```

**good for:** “distribution” outputs; KL-divergence losses; constrained learning.

---

# Group E — Permutation & order structure (stress tests)

## 17) `FixedPermutationDataset`

**math:** ( y = P x ) where (P) is a fixed permutation
**description:** deterministic shuffle/unscramble task.
**numpy implementation:**

```python
perm = rng.permutation(self.d)
X = (rng.random((self.num_samples, self.d)) * 2 - 1).astype(float)
Y = X[:, perm]
```

**good for:** testing whether your architecture can learn index re-mapping.

---

## 18) `SortDataset`

**math:** ( y = \text{sort}(x) ) (ascending)
**description:** non-smooth, highly structured mapping.
**numpy implementation:**

```python
X = (rng.random((self.num_samples, self.d)) * 2 - 1).astype(float)
Y = np.sort(X, axis=1)
```

**good for:** hard-mode; shows limits of standard feedforward nets quickly.

---

# Group F — “Not regression” but still dim-preserving

## 19) `BinaryVectorClassificationDataset`

**math:** ( y \in {0,1}^d ) with (y_i = \mathbb{1}[x_i > t])
**description:** per-dimension binary labels.
**numpy implementation:**

```python
X = (rng.standard_normal((self.num_samples, self.d))).astype(float)
Y = (X > self.threshold).astype(float)  # float for BCE
```

**good for:** sigmoid output + BCE; calibration; thresholding tasks.

---

## 20) `MultiLabelFromLinearDataset`

**math:** ( y = \mathbb{1}[Ax + b > 0] )
**description:** coupled multi-label classification with same dimension.
**numpy implementation:**

```python
X = rng.standard_normal((self.num_samples, self.d)).astype(float)
A = rng.standard_normal((self.d, self.d)).astype(float)
b = rng.standard_normal((self.d,)).astype(float) * self.b_scale
logits = X @ A.T + b
Y = (logits > 0).astype(float)
```

**good for:** learning correlated label structure; BCE vs hinge-style losses (if added).

---

## 21) `BitFlipDenoiseDataset`

**math:** input bits flipped with prob p; target is original bits
**description:** discrete denoising / error-correcting flavor.
**numpy implementation:**

```python
Y = (rng.random((self.num_samples, self.d)) > 0.5).astype(float)
X = Y.copy()
flip = rng.random(X.shape) < self.p
X[flip] = 1.0 - X[flip]
```

**good for:** BCE; robustness; whether the net learns parity-ish corrections.

---

# Group G — Sparsity & mixture structure

## 22) `SparseIdentityDataset`

**math:** ( y = x ) but (x) sparse
**description:** most dimensions are zero; nonzeros carry signal.
**numpy implementation:**

```python
X = np.zeros((self.num_samples, self.d), dtype=float)
k = self.k_nonzero
idx = np.stack([rng.choice(self.d, size=k, replace=False) for _ in range(self.num_samples)])
vals = rng.standard_normal((self.num_samples, k))
X[np.arange(self.num_samples)[:, None], idx] = vals
Y = X.copy()
```

**good for:** sparse regimes; Adam vs SGD; handling many zeros.

---

## 23) `LowRankLinearDataset`

**math:** ( y = U V^\top x ) (rank r)
**description:** mixing but constrained rank; easier structure than full dense.
**numpy implementation:**

```python
X = rng.standard_normal((self.num_samples, self.d)).astype(float)
U = rng.standard_normal((self.d, self.r)).astype(float)
V = rng.standard_normal((self.d, self.r)).astype(float)
A = U @ V.T
Y = X @ A.T
```

**good for:** structured coupling; generalization vs memorization.

---

# What to add to your activation/loss menu for these

## Recommended output activations

* **Linear**: most of these (default)
* **Sigmoid**: datasets 19–21 (binary/multi-label)
* **Tanh**: bounded symmetric outputs (good with sine/power)
* **Softplus/ReLU**: nonnegative projection tasks (15–16)

## Recommended losses to add (high ROI)

* **L1 / MAE** (especially for denoise/outliers)
* **BCE** (for binary vector tasks)
* **Cosine distance** (for unit-sphere / rotation tasks)
* **KL divergence** (for simplex/probability outputs)

---

If you tell me whether you’re targeting **PyTorch** or **pure NumPy**, I can give you a clean class template (base `Dataset`, `make_batch()`, parameter serialization) that makes these plug-and-play in your CLI.
