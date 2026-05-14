# HexNets web UI (Streamlit)

Python package: [`hexnets_web`](../src/hexnets_web/). Entry script: [`src/streamlit_app.py`](../src/streamlit_app.py).

## Summary (for quick orientation)

* **Entry:** `src/streamlit_app.py` — launch: `make streamlit-run` or `streamlit run src/streamlit_app.py`.
* **Navigation:** `hexnets_web/main.py` uses `st.navigation(..., position="sidebar")` and `st.Page` scripts under `src/hexnets_web/pages/` (paths relative to `streamlit_app.py`). Default landing page: **CLI Builder**. Each page implements `render()` on a subclass of `hexnets_web.pages.base_page.BasePage`. Optional **Buy Me a Coffee** (or any third-party `<script>` widget): set `_BUY_ME_A_COFFEE_HTML` in `main.py` and render it with `streamlit.components.v1.html` in `st.sidebar` after `pg.run()` (markdown strips scripts).
* **Pages:** **CLI Builder** (argparse-driven command preview + inline glossary for selected registry keys; **global** argparse group = right-hand column: `--model`, `--seed`, `--activation`, `--loss`, `-lr` / `--learning-rate`; **training** and other groups = left-hand command options), **Network Explorer** (live `HexagonalNeuralNetwork`; `st.columns(2)`: geometry + **Generate Structure Graph** left, structure summary right; session still holds activation / loss / learning-rate defaults used when the network is built — hyperparameter dropdowns are mostly commented out in code), **Rotation Comparison** (25/75 layout: sliders + multi-activation left, three reference images right; needs `hexnet ref --all` for full grid), **Lesion Lab** (placeholder), **Run Browser** (`st.columns([1.15, 1, 1])`: left — Family A tree + session paths; two **Compare** columns with `config.json` + `plots/*net_training_*.png`), **Dataset Generator** (Dataset / Noise tabs; cached clean `X` via `configure_data`, **RNG** vs **UNIFORM** input sampling, **Regenerate Inputs**, noise toggles, two ECharts scatters), **Glossary** (searchable nested terms; tree in `src/hexnets_web/pages/glossary/glossary_data.py`, node type in `glossary_types.py`; top-level branches from `build_*_glossary_parent()` in `src/data/dataset.py`, `src/networks/loss/loss.py`, `src/networks/learning_rate/learning_rate.py`, `src/networks/activation/activations.py`; UI in `pages/glossary/glossary.py` / `metrics_explainer.py`).
* **Rotation Comparison layout:** `st.columns([1, 3])` — left: `n` slider, multi-activation (`hexnet_n{n}_multi_activation.png`) under `n`, then `r` slider; right: three columns for structure, activation, and weight for `(n, r)`.
* **Defaults:** Network Explorer: `n=2`, `r=0`, `activation=relu`, `loss=mean_squared_error`, `learning_rate=constant`, `dataset_type=identity`, `dataset_num_samples=100`. Rotation Comparison: `rotation_comparison_n=2`, `rotation_comparison_r=0` (see `initialize_session_state()`).

## Overview

The HexNets Streamlit application provides an interactive web interface for visualizing and exploring hexagonal neural networks. It offers seven sidebar pages (CLI Builder first / default):

1. **CLI Builder**: Build and copy `hexnet …` commands from live controls
2. **Network Explorer**: Two columns — geometry (`n`, `r`) and **Generate Structure Graph**; opposite column shows live structure metrics (see **Network Explorer page** below)
3. **Rotation Comparison**: Sliders for `n` and `r` to browse pre-generated reference images for one `(n, r)` at a time (scrub `r` to compare rotations)
4. **Lesion Lab**: Placeholder for future experiments
5. **Run Browser**: `runs/` root; Family A autodiscover + session-added paths; two compare columns (`config.json` + training PNGs per slot)
6. **Dataset Generator**: **Dataset** / **Noise** tabs (Noise unused for now). **Dataset**: glossary + **Regenerate Inputs**; **Options** (samples, **Input sampling** `RNG` vs `UNIFORM`, Gaussian noise). Builds `BaseDataset` via `build_registered_dataset()` then **`configure_data`** with session-stored clean `X` so inputs stay fixed while noise / sampling mode changes invalidate via cache key `(dataset, d, num_samples, mode)`. Side-by-side **Input** / **Outputs** ECharts scatters; fixed `d=2` in page code.
7. **Glossary**: Filterable definitions (including nested entries) aligned with metrics and datasets used in the app

## Architecture

### Application Structure

The Streamlit app is launched from `src/streamlit_app.py` and implemented under `src/hexnets_web/`: `main.py` registers `st.Page` routes; each page lives under `hexnets_web/pages/<area>/` (thin `*_page.py` entry + `BasePage` subclass, except glossary uses `glossary.py` + `glossary_page.py`). Shared helpers at package root: `session.py`, `figures.py`, `references.py`, `metrics_explainer.py`, `glossary_types.py`, `cli_types.py`.

### Key Components

#### 1. Session State Management
```python
def initialize_session_state():
    # Stores network parameters (n, r, activation, loss, learning_rate), dataset picks,
    # rotation-comparison viewer keys (rotation_comparison_n / rotation_comparison_r),
    # and a cached network instance
```

The app uses Streamlit's session state to:
- Persist user selections across interactions
- Cache the network instance to avoid unnecessary recreation
- Maintain parameter consistency between UI controls and network generation

#### 2. Matplotlib Figure Conversion
```python
def create_matplotlib_figure(fig):
    # Converts matplotlib figure to BytesIO buffer
    # Returns buffer suitable for Streamlit's st.image()
```

Since Streamlit doesn't natively display matplotlib figures, the app converts them to PNG images in memory using `BytesIO` buffers.

#### 3. Reference Image Loading
```python
def load_reference_image(n, r, image_type):
    # Loads pre-generated reference images from reference/ directory
    # Supports: structure, activation, weight image types
```

The Rotation Comparison page relies on pre-generated reference images stored in the `reference/` directory.

## Generating Reference Images

### Prerequisites

Before the Streamlit app can fully function (especially the Rotation Comparison page), reference images must be generated.

### Generation Process

Reference images are created using the `hexnet ref` command:

```bash
# Generate all reference graphs for n=2..8 and r=0..5
hexnet ref --all
```

This command:
1. Creates a `reference/` directory (if it doesn't exist)
2. Generates graphs for all combinations of:
   - **n values**: 2 through 8 (7 total)
   - **r values**: 0 through 5 (6 rotations)
3. Produces three types of graphs per (n, r) combination:
   - **Structure graphs**: Physical network layout (`hexnet_n{n}_r{r}_structure.png`)
   - **Activation matrices**: Binary activation patterns (`hexnet_n{n}_r{r}_Activation_Structure.png`)
   - **Weight matrices**: Full weight matrix visualizations (`hexnet_n{n}_r{r}_Weight_Matrix.png`)
4. Generates multi-activation overlays (one per n value)

**Total output**: 133 reference files
- Structure graphs: 6 rotations × 7 n values = 42 files
- Activation matrices: 6 rotations × 7 n values = 42 files
- Weight matrices: 6 rotations × 7 n values = 42 files
- Multi-activation overlays: 7 n values = 7 files

### File Naming Convention

Reference images follow a consistent naming pattern:
- Structure: `hexnet_n{n}_r{r}_structure.png`
- Activation: `hexnet_n{n}_r{r}_Activation_Structure.png` (title "Activation Structure" with spaces replaced by underscores)
- Weight: `hexnet_n{n}_r{r}_Weight_Matrix.png` (title "Weight Matrix" with spaces replaced by underscores)
- Multi-activation: `hexnet_n{n}_multi_activation.png`

**Note**: The activation and weight filenames are generated by replacing spaces in the title with underscores. The Streamlit app's `load_reference_image()` function matches these exact patterns.

### Generation Details

The `--all` flag in the reference command:
- Ignores individual `-n` and `-r` arguments
- Uses memoization to avoid recreating networks for the same rotation
- Provides progress feedback with colored output
- Handles errors gracefully, continuing with other combinations if one fails

## Running the Streamlit App

### Local Development

#### Prerequisites
1. Virtual environment with dependencies installed:
   ```bash
   make install
   ```

2. (Optional) Reference images generated:
   ```bash
   hexnet ref --all
   ```
   *Note: The app will work without reference images, but the Rotation Comparison page will show warnings.*

#### Launch Methods

**Using Makefile (Recommended)**:
```bash
make streamlit-run
```

**Direct Streamlit Command**:
```bash
streamlit run src/streamlit_app.py
```

**With Virtual Environment**:
```bash
source .venv/bin/activate
streamlit run src/streamlit_app.py
```

### Application Behavior

When launched, the Streamlit app:
1. Initializes session state with default values (n=2, r=0, activation=relu, loss=mean_squared_error, learning_rate=constant, dataset registry fields, rotation-comparison sliders)
2. Creates a network instance and caches it in session state
3. Displays sidebar navigation with seven pages (default **CLI Builder**):
   - **CLI Builder**: Subcommand pickers, preview, inline glossary for applicable options
   - **Network Explorer**: Interactive controls and on-demand graph generation
   - **Rotation Comparison**: Pre-generated reference image viewer (25/75 layout)
   - **Lesion Lab**: Coming soon placeholder
   - **Run Browser**: `st.columns([1.15, 1, 1])` — left: Family A tree + session-added paths; right: two **Compare** slots (`config.json` + training PNGs each), assign via **→1 / →2** or per-column `st.selectbox`
   - **Dataset Generator**: Dataset/Noise tabs; `configure_data` + cached `X`, RNG/UNIFORM sampling, Input/Outputs scatters (`pages/dataset_generator/dataset_generator.py`)
   - **Glossary**: Search field filters top-level and nested glossary entries (substring match, case-insensitive); top-level expanders are shown in two columns when multiple entries match

### Network Explorer page

**Features**:
- **Layout** (`st.columns(2)`): **Left — Geometry & data:** `n` (`st.number_input`, 2–25), `r` (`st.slider`, 0–5), **Generate Structure Graph** (renders `graph_structure` inline). Dataset / activation / loss / learning-rate **selectboxes** exist in code but are **commented out**; defaults still come from `initialize_session_state()` (`learning_rate` uses the same registry as CLI when reconciling invalid session values). **Right — Information:** `update_network()` then metrics (total nodes, layers, edges), layer-size caption, per-layer index lists (preview truncated for long lists).
- **Actions**: **Generate Structure Graph** — refreshes the live network and shows one matplotlib structure figure.

- **Session / network sync:** The info column calls `update_network()` each run so layer indices and sizes match the current `n`, `r`, activation, loss, and learning rate from session state.

**Graph Generation**:
- Graphs are generated dynamically using `HexagonalNetwork` methods
- Figures are converted to images and displayed inline
- Generated images are saved to `reference/` directory
- Figures are properly closed after display to prevent memory leaks

### Rotation Comparison page

**Features**:
- Sliders for `n` (2–8) and `r` (0–5), same ranges as Network Explorer; values live in `st.session_state.rotation_comparison_n` / `rotation_comparison_r` so they do not change the live network.s `n`/`r`.
- **Layout** (`st.columns([1, 3])`): narrow column has `n` slider, multi-activation image (per `n` only), then `r` slider; wide column has three images for the selected `(n, r)` — physical structure, activation pattern, weight matrix.

**Image Loading**:
- Attempts to load images from `reference/` directory
- Shows warnings if images are missing
- Gracefully handles missing files without crashing

### Run Browser page

- **Runs root:** `pathlib.Path("runs/").resolve()` (matches `RunService.runs_dir` when the app is started from the repo root). UI: `src/hexnets_web/pages/run_browser/run_browser.py`; validation: `run_validation.py`; Family A dirname parsing / pair index: `fam_a_signatures.py`; plot discovery: `training_plots.py` (`_training_plot_paths`).
- **Tabs:** `st.tabs` — **Family A picker** vs **Manual compare** (session keys for manual compare are shared across tabs so assignments persist when switching tabs).
- **Family A picker tab:** Scans `runs/e2etest-famA/` for valid runs whose folder names match the e2e layout from [`scripts/e2e-bench-A.sh`](../scripts/e2e-bench-A.sh): `hex|mlp-<dataset>-<activation>-<loss>-<learning_rate>-n{dims}`. Dependent **Dataset → Activation → Loss → Learning rate → Dimensions** (`n…` suffix) selectboxes list only values that exist on disk for the current prefix. **Random** picks a random indexed signature (may be hex-only, mlp-only, or a full pair). **Hex** and **MLP** columns are fixed (left hex, right mlp); a missing side shows an info message. A short caption notes full vs partial pair on disk. Each side shows `config.json` (`st.json` / raw fallback) and training PNGs; if there are no `plots/*net_training_*.png` files, an explicit **info** message is shown.
- **Manual compare tab:** `st.columns([0.75, 1, 1])` — narrow **Runs tree**; two **Compare** columns side by side.
- **Valid run:** A directory must contain `config.json`, `manifest.json`, and `training_metrics.json` at its root (`is_valid_run_dir` / `missing_run_artifacts`). Only valid runs appear under Family A or can be added manually.
- **Family A tree (manual tab):** Under `runs/e2etest-famA/`, immediate subfolders that are valid runs are grouped by **class** (leading segment of the folder name before the first `-`, e.g. `hex-…` vs `mlp-…`; other names bucket separately). All valid runs are listed inside each class expander; the first two class expanders default to **expanded**. Each run row has **→1 / →2** buttons (`st.button(..., on_click=_assign_run_to_slot, kwargs=…)`) so the slot updates before the next draw.
- **Add run by path:** Text field + **Add run** (no native OS folder dialog in Streamlit). Path is resolved with `pathlib.Path(…).expanduser().resolve()`; if valid, it is appended to `st.session_state.run_browser_session_paths` (session-only; lost on refresh). **Remove** drops a session entry and clears any compare slot pointing at it.
- **Compare columns (2, manual tab):** Each has a **Run** `st.selectbox` over the union of Family A paths plus session-added paths (plus `(empty)`). Empty slot: info prompt. Filled slot: `config.json` only via `st.json` (invalid JSON falls back to `st.code`); below that, every `plots/*net_training_*.png` via `st.image`. The same path may fill both slots.

### Lesion Lab page

- Placeholder: **Coming soon** (scope undetermined).

### Glossary page

- **Search**: `st.text_input` with case-insensitive substring filtering. Each entry.s index includes its nested children so terms like “identity” match under **Datasets**.
- **Layout**: Top-level entries use two columns of expanders when multiple roots are visible; nested definitions stay inside their parent expander.
- **Content**: Plain-language explanations with optional `st.latex` formulas and examples; glossary tree in `src/hexnets_web/pages/glossary/glossary_data.py` (`GlossaryNode` in `glossary_types.py`). Registered **datasets**, **losses**, **learning rates**, and **activations** each expose `get_glossary_node()` on the class and a hub `build_*_glossary_parent()` (datasets: `src/data/dataset.py`; losses / learning rates / activations: hub files under `src/networks/loss/`, `src/networks/learning_rate/`, `src/networks/activation/`). Page UI: `pages/glossary/glossary.py` (`GlossaryPage`).

## Deployment to Streamlit Cloud

### Prerequisites

1. **Code Repository**: Code must be pushed to a GitHub repository
2. **Requirements File**: Either `requirements.txt` or `pyproject.toml` must exist
3. **Reference Images** (Optional): For full functionality, reference images should be generated

### Deployment Preparation

Check deployment readiness:
```bash
make streamlit-deploy
```

This command:
- Validates that `src/streamlit_app.py` exists
- Checks for `requirements.txt` (warns if missing)
- Verifies `reference/` directory exists (warns if missing)
- Provides step-by-step deployment instructions

### Deployment Steps

1. **Push code to GitHub**: Ensure all code is committed and pushed
2. **Visit Streamlit Cloud**: Go to https://share.streamlit.io/
3. **Sign in**: Use your GitHub account
4. **Create New App**: Click "New app" and select your repository
5. **Configure**:
   - **Main file path**: `src/streamlit_app.py`
   - **Python version**: 3.9 or higher
   - **Requirements file**: `requirements.txt` (or `pyproject.toml`)
6. **Deploy**: Click "Deploy!"

### Deployment Considerations

**Reference Images**:
- Reference images are not automatically generated on Streamlit Cloud
- Options:
  1. Commit reference images to the repository (large files, not recommended)
  2. Generate them as part of the deployment process (requires custom setup)
  3. Accept that Rotation Comparison page will show warnings

**Dependencies**:
- Streamlit Cloud automatically installs dependencies from `requirements.txt` or `pyproject.toml`
- Ensure all required packages are listed

**File Paths**:
- The app uses relative paths (`./reference/`)
- These resolve correctly in Streamlit Cloud's environment

## Technical Details

### Figure Management

The app properly manages matplotlib figures to prevent memory leaks:

```python
# Generate figure
filename, fig = net.graph_structure(output_dir=streamlit_dir, medium="matplotlib")

# Convert to image
buf = create_matplotlib_figure(fig)

# Display
st.image(buf, use_container_width=True)

# Clean up
plt.close(fig)
```

### Output Directory

The app uses a consistent output directory:
- **Local development**: `./reference/` (relative to project root)
- **Streamlit Cloud**: Same relative path (resolves in deployment environment)

### Network Instance Lifecycle

1. **Initialization**: Network created on first load with default parameters (`initialize_session_state()` + `update_network()` as needed).
2. **Update (Network Explorer):** `st.session_state.net` is rebuilt when controls change so the structure column stays consistent with `n` / `r` / activation / loss / learning rate; **Generate Graphs** refreshes the network again before plotting.
3. **Caching**: The live instance is held in `st.session_state.net` between reruns.

### Error Handling

The app includes error handling for:
- Missing reference images (shows warnings, doesn't crash)
- Network generation failures (displays error messages)
- Invalid parameter combinations (handled by validation in HexagonalNetwork)

## Troubleshooting

### Issue: Rotation Comparison shows warnings

**Solution**: Generate reference images:
```bash
hexnet ref --all
```

### Issue: Graphs not displaying

**Possible causes**:
1. Network generation failed (check error messages)
2. Matplotlib backend issues (should work automatically with Streamlit)
3. Memory issues (try restarting the app)

### Issue: App won't start

**Check**:
1. Virtual environment is activated
2. Dependencies are installed: `make install`
3. Streamlit is installed: `pip install streamlit`
4. Python version is 3.9+

### Issue: Changes not reflecting

**Solution**: Streamlit has a "Rerun" button, or save the file to trigger auto-reload

## Future Enhancements

Potential improvements:
- Externalize glossary entries (e.g. YAML) for easier editing without code changes
- Generate reference images on-demand if missing
- Add more visualization types
- Support for custom network configurations
- Export functionality for generated graphs
- Training history visualization
- Interactive weight editing
