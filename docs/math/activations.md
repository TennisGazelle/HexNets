Assumptions in snippets:

* `x` = input tensor/array
* All activations implement `activate(x)` and `deactivate(x)` methods
* `deactivate(x)` returns the derivative/gradient of the activation at `x`

---

# Activation Functions

## 1) `Linear`

**math:** ( f(x) = x )
**description:** identity/passthrough activation; no nonlinearity.
**numpy implementation:**

```python
def activate(self, x):
    return x

def deactivate(self, x):
    return 1.0
```

**good for:** output layers; linear regression; debugging; baseline comparisons.

---

## 2) `ReLU`

**math:** ( f(x) = \max(0, x) )
**description:** rectified linear unit; zero for negative inputs, identity for positive.
**numpy implementation:**

```python
def activate(self, x):
    return np.maximum(0, x)

def deactivate(self, x):
    return (x > 0).astype(float)
```

**good for:** hidden layers; sparse activations; common default choice; nonnegative outputs.

---

## 3) `LeakyReLU`

**math:** ( f(x) = \begin{cases} x & \text{if } x > 0 \\ \alpha x & \text{if } x \leq 0 \end{cases} )
**description:** leaky rectified linear unit; allows small negative gradients.
**parameters:**
* `alpha` (default: 0.01): slope for negative inputs

**numpy implementation:**

```python
def __init__(self, alpha=0.01):
    self.alpha = alpha

def activate(self, x):
    return np.where(x > 0, x, self.alpha * x)

def deactivate(self, x):
    return np.where(x > 0, 1, self.alpha)
```

**good for:** preventing "dying ReLU" problem; maintaining gradient flow for negative inputs.

---

## 4) `Sigmoid`

**math:** ( f(x) = \frac{1}{1 + e^{-x}} )
**description:** sigmoid/logistic function; maps inputs to (0, 1).
**numpy implementation:**

```python
def activate(self, x):
    return 1 / (1 + np.exp(-x))

def deactivate(self, x):
    return self.activate(x) * (1 - self.activate(x))
```

**good for:** output layers for binary classification; probability outputs; bounded outputs in (0, 1).

---
