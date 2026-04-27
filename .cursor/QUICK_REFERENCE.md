# HexNets Quick Reference

## Running the Streamlit Web Interface

### Launch Streamlit Locally
```bash
# Using Makefile
make run-streamlit

# Or directly
streamlit run src/streamlit_app.py
```

The Streamlit app provides:
- **Network Explorer**: Interactive visualization with parameter controls
- **Rotation Comparison**: Table view of all 6 rotations (similar to ROTATION_SYSTEM.md)

**Note:** The Rotation Comparison tab requires reference graphs to be generated first:
```bash
hexnet ref --all
```

### Deploy to Streamlit Cloud
```bash
# Check deployment readiness and get instructions
make streamlit-deploy
```

This will:
- Validate prerequisites (app file, requirements.txt)
- Provide step-by-step deployment instructions
- Guide you through the Streamlit Community Cloud setup

**Deployment requires:**
- Code pushed to a GitHub repository
- `requirements.txt` file (or `pyproject.toml` will work)
- Access to https://share.streamlit.io/

## CLI Commands

### Generate All Reference Graphs
```bash
hexnet ref --all
```

**Note:** The `--all` flag generates all reference graphs for n=2..8 and r=0..5. Other arguments like `-n` and `-r` are ignored when `--all` is used.

### Generate Single Reference Graph
```bash
hexnet ref -n 3 -r 0 -g structure_matplotlib
```

### Train a Network
```bash
# After `make install` (venv active): console script from pyproject.toml
hexnet train -m hex -n 3 -r 0 -e 100 -l mean_squared_error -a sigmoid -t identity
```

### Run statistics on a saved run
```bash
hexnet stats runs/<run_name>
```

### List Available Components
```python
from networks.activation.activations import get_available_activation_functions
from networks.loss.loss import get_available_loss_functions
from networks.learning_rate.learning_rate import get_available_learning_rates

print(get_available_activation_functions())
print(get_available_loss_functions())
print(get_available_learning_rates())
```

### Iterate All Loss Functions
```bash
# From repo root with `.venv` on PATH (e.g. `source .venv/bin/activate`)
for loss in $(python -c "from networks.loss.loss import get_available_loss_functions; print(' '.join(get_available_loss_functions()))"); do
    hexnet train -m hex -n 3 -r 0 -e 50 -l "$loss" -t identity -rn "loss_${loss}"
done
```

### Iterate All Activation Functions
```bash
for act in $(python -c "from networks.activation.activations import get_available_activation_functions; print(' '.join(get_available_activation_functions()))"); do
    hexnet train -m hex -n 3 -r 0 -e 50 -a "$act" -t identity -rn "act_${act}"
done
```

### Iterate All Combinations
```python
from itertools import product
from networks.activation.activations import get_available_activation_functions
from networks.loss.loss import get_available_loss_functions
import subprocess

for loss, activation in product(
    get_available_loss_functions(),
    get_available_activation_functions()
):
    subprocess.run([
        "hexnet", "train", "-m", "hex", "-n", "3", "-r", "0", "-e", "50",
        "-l", loss, "-a", activation, "-t", "identity",
        "-rn", f"combo_{loss}_{activation}",
    ])
```

## Component Locations

- **Loss Functions:** `src/networks/loss/*.py`
- **Activations:** `src/networks/activation/*.py`
- **Learning Rates:** `src/networks/learning_rate/*.py`
- **Datasets:** `src/data/*.py`
- **Networks:** `src/networks/*.py`

**For adding new components, see [COMPONENT_DEVELOPMENT.md](./COMPONENT_DEVELOPMENT.md)**

## CLI Argument Reference

| Argument | Short | Description | Default |
|----------|-------|-------------|---------|
| `--num_dims` | `-n` | Input/output dimensions | 3 |
| `--rotation` | `-r` | Hex rotation (0-5) | 0 |
| `--model` | `-m` | Model type (hex/mlp) | hex |
| `--activation` | `-a` | Activation function | sigmoid |
| `--loss` | `-l` | Loss function | mean_squared_error |
| `--learning-rate` | `-lr` | Learning rate schedule | constant |
| `--epochs` | `-e` | Number of epochs | 100 |
| `--pause` | `-p` | Pause between epochs | 0.05 |
| `--type` | `-t` | Dataset type | identity |
| `--dataset-size` | `-ds` | Number of samples | 250 |
| `--seed` | `-s` | Random seed | 42 |
| `--run_name` | `-rn` | Custom run name | auto |
| `--run-dir` | `-rd` | Load existing run | None |
