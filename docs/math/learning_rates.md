Assumptions in snippets:

* `iteration` = current training iteration (0-indexed or 1-indexed, depending on implementation)
* All learning rate schedulers implement `rate_at_iteration(iteration)` method
* Returns a scalar learning rate value (typically > 0)

---

# Learning Rate Schedulers

## 1) `ConstantLearningRate`

**math:** ( \eta(t) = \eta_0 )
**description:** constant learning rate; no decay or adaptation over time.
**parameters:**
* `learning_rate` (default: 0.01): fixed learning rate value

**numpy implementation:**

```python
def __init__(self, learning_rate: float = 0.01):
    self.learning_rate = learning_rate

def rate_at_iteration(self, iteration: int) -> float:
    return self.learning_rate
```

**good for:** simple baselines; debugging; when manual tuning is preferred; stable optimization problems.

---

## 2) `ExponentialDecayLearningRate`

**math:** ( \eta(t) = \eta_0 \cdot \gamma^t )
**description:** exponential decay learning rate; multiplies by decay rate each iteration.
**parameters:**
* `initial_learning_rate` (default: 0.01): starting learning rate value
* `decay_rate` (default: 0.95): multiplicative decay factor per iteration (typically 0 < γ < 1)

**numpy implementation:**

```python
def __init__(self, initial_learning_rate: float = 0.01, decay_rate: float = 0.95):
    self.initial_learning_rate = initial_learning_rate
    self.decay_rate = decay_rate

def rate_at_iteration(self, iteration: int) -> float:
    return self.initial_learning_rate * (self.decay_rate ** iteration)
```

**good for:** gradual learning rate reduction; fine-tuning convergence; preventing oscillations; adaptive optimization schedules.

---
