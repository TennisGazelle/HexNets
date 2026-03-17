# Component Development Guide

This guide explains how to add new components to the HexNets framework.

## Component Registration System

All components use Python's `__init_subclass__` hook to automatically register themselves when the class is defined. This means:

1. **No manual registration needed** - Just create the class and it's available
2. **Auto-discovery** - Components are found by scanning directories
3. **Display names** - Human-readable names separate from class names

## Adding a New Loss Function

### Step 1: Create the File
Create `src/networks/loss/MyLoss.py`:

```python
import numpy as np
from networks.loss.loss import BaseLoss

class MyLoss(BaseLoss, display_name="my_loss"):
    """
    Your loss function description here.
    """
    
    def calc_loss(self, y_true, y_pred):
        """
        Compute the loss value.
        
        Args:
            y_true: Ground truth values (numpy array)
            y_pred: Predicted values (numpy array)
        
        Returns:
            Scalar loss value
        """
        # Example: Mean Absolute Error
        return np.mean(np.abs(y_true - y_pred))
    
    def calc_delta(self, y_true, y_pred):
        """
        Compute the gradient of loss w.r.t. y_pred.
        This is used in backpropagation.
        
        Args:
            y_true: Ground truth values (numpy array)
            y_pred: Predicted values (numpy array)
        
        Returns:
            Gradient array (same shape as y_pred)
        """
        # Gradient of MAE
        signs = np.sign(y_pred - y_true)
        return signs / y_true.shape[0]
```

### Step 2: Ensure Module is Imported
The `__init__.py` in `src/networks/loss/` automatically imports all `.py` files. Just make sure your file follows the naming convention (not `__init__.py` or `loss.py`).

### Step 3: Test It
```python
from networks.loss.loss import get_loss_function, get_available_loss_functions

# Should now include "my_loss"
print(get_available_loss_functions())

# Can instantiate it
loss = get_loss_function("my_loss")
```

### Step 4: Use in CLI
```bash
python -m src.cli train -l my_loss -n 3 -e 50
```

## Adding a New Activation Function

### Step 1: Create the File
Create `src/networks/activation/MyActivation.py`:

```python
import numpy as np
from networks.activation.activations import BaseActivation

class MyActivation(BaseActivation, display_name="my_activation"):
    """
    Your activation function description.
    """
    
    def activate(self, x):
        """
        Apply activation function element-wise.
        
        Args:
            x: Input array (numpy array)
        
        Returns:
            Activated values (same shape as x)
        """
        # Example: Swish activation
        return x * (1 / (1 + np.exp(-x)))
    
    def deactivate(self, x):
        """
        Compute derivative of activation function.
        Used in backpropagation.
        
        Args:
            x: Input array (numpy array) - typically pre-activation values
        
        Returns:
            Derivative values (same shape as x)
        """
        # Derivative of Swish
        sigmoid = 1 / (1 + np.exp(-x))
        return sigmoid * (1 + x * (1 - sigmoid))
```

### Step 2: Test It
```python
from networks.activation.activations import get_activation_function

act = get_activation_function("my_activation")
x = np.array([-1, 0, 1])
print(act.activate(x))
print(act.deactivate(x))
```

## Adding a New Learning Rate Schedule

### Step 1: Create the File
Create `src/networks/learning_rate/ExponentialDecay.py`:

```python
from networks.learning_rate.learning_rate import BaseLearningRate

class ExponentialDecay(BaseLearningRate, display_name="exponential_decay"):
    """
    Exponential decay learning rate schedule.
    """
    
    def __init__(self, initial_rate=0.01, decay_rate=0.95):
        self.initial_rate = initial_rate
        self.decay_rate = decay_rate
    
    def rate_at_iteration(self, iteration: int) -> float:
        """
        Get learning rate for a given iteration.
        
        Args:
            iteration: Current iteration number (0-indexed)
        
        Returns:
            Learning rate value
        """
        return self.initial_rate * (self.decay_rate ** iteration)
```

### Step 2: Update Factory Function (if needed)
If your learning rate needs constructor arguments, you may need to handle them in the factory. Check `get_learning_rate()` in `learning_rate.py`.

## Adding a New Dataset

### Step 1: Create the File
Create `src/data/MyDataset.py`:

```python
import numpy as np
from data.dataset import BaseDataset

class MyDataset(BaseDataset, display_name="my_dataset"):
    """
    Your dataset description.
    """
    
    def __init__(self, d: int = 2, num_samples: int = 100, **kwargs):
        super().__init__()
        self.d = d
        self.num_samples = num_samples
        self.kwargs = kwargs
        self.load_data()
    
    def load_data(self) -> bool:
        """
        Load or generate the dataset.
        
        Returns:
            True if successful
        """
        # Generate random inputs
        X = (np.random.rand(self.num_samples, self.d) * 2 - 1).astype(float)
        
        # Your transformation here
        Y = X ** 2  # Example: square transformation
        
        self.data = {
            "X": X,
            "Y": Y,
        }
        return True
```

### Step 2: Update Dataset Factory (if needed)
Update `get_dataset()` in `src/commands/command.py` to include your new dataset type.

## Adding a New Network Type

### Step 1: Create the Network Class
Create `src/networks/MyNetwork.py`:

```python
import numpy as np
from networks.network import BaseNeuralNetwork
from networks.activation.activations import BaseActivation
from networks.loss.loss import BaseLoss

class MyNetwork(BaseNeuralNetwork):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        activation: BaseActivation,
        loss: BaseLoss,
        learning_rate=0.01
    ):
        super().__init__(learning_rate, activation, loss)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        # Initialize weights, etc.
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        # Implement forward pass
        pass
    
    def backward(self, activations, target, apply_delta_W=True):
        # Implement backward pass
        pass
    
    # ... implement all abstract methods
```

### Step 2: Update RunService
Add your network type to `RunService.__init__()` in `src/services/run_service/RunService.py`:

```python
elif args.model == "my_network":
    self.net = MyNetwork(
        input_dim=args.input_dim,
        # ... other parameters
    )
```

### Step 3: Update CLI
Add your model to the choices in `add_global_arguments()` in `src/commands/command.py`:

```python
parser.add_argument(
    "-m", "--model",
    choices=["hex", "mlp", "my_network"],  # Add here
    ...
)
```

## Testing Your Component

### Unit Test Template
```python
import numpy as np
from networks.loss.loss import get_loss_function

def test_my_loss():
    loss = get_loss_function("my_loss")
    
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.1, 1.9, 3.1])
    
    loss_value = loss.calc_loss(y_true, y_pred)
    delta = loss.calc_delta(y_true, y_pred)
    
    assert loss_value > 0
    assert delta.shape == y_pred.shape
    print("✓ Loss function works!")
```

### Integration Test
```python
from networks.HexagonalNetwork import HexagonalNeuralNetwork
from networks.activation.activations import get_activation_function
from networks.loss.loss import get_loss_function
from data.dataset import IdentityDataset

# Test with your component
loss = get_loss_function("my_loss")
activation = get_activation_function("sigmoid")

net = HexagonalNeuralNetwork(
    n=3,
    activation=activation,
    loss=loss
)

data = IdentityDataset(d=3, num_samples=100)
net.train(data, epochs=10)
print("✓ Integration test passed!")
```

## Best Practices

1. **Follow Naming Conventions**
   - Class names: `PascalCase`
   - Display names: `snake_case`
   - File names: Match class name

2. **Documentation**
   - Add docstrings to all methods
   - Explain mathematical operations
   - Document expected input/output shapes

3. **Error Handling**
   - Validate inputs (shapes, ranges)
   - Provide clear error messages
   - Handle edge cases (e.g., division by zero)

4. **Numerical Stability**
   - Use `np.clip()` for activations that can overflow
   - Add small epsilon values where needed
   - Consider using `np.float64` for precision

5. **Testing**
   - Test with known inputs/outputs
   - Test edge cases (zeros, negatives, large values)
   - Verify gradients are correct (if applicable)

## Common Pitfalls

1. **Forgetting `display_name`** - Component won't be discoverable
2. **Wrong method signatures** - Must match base class exactly
3. **Not importing the module** - Component won't register
4. **Shape mismatches** - Ensure arrays have compatible shapes
5. **Gradient errors** - `calc_delta` must be correct for training to work
