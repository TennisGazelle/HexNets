# Reference Files Catalog

This document catalogs all reference graph files that should be generated for the hexagonal network system. Reference graphs are structure visualizations that do not depend on training (they show the network topology, not learned weights).

## File Naming Convention

Reference files are saved to the `reference/` directory and follow these patterns:

- **Structure graphs:** `hexnet_n{n}_r{r}_structure.png`
- **Activation matrices:** `hexnet_n{n}_r{r}_Activation_Structure.png`
- **Weight matrices:** `hexnet_n{n}_r{r}_Weight_Matrix.png`
- **Multi-rotation overlay:** `hexnet_n{n}_multi_activation.png`
- **Layer indices (terminal):** Output to console, not saved

## Graph Types

### 1. Structure Graphs (`structure_matplotlib`)

Visual representation of the network graph showing:
- Node positions arranged in layers
- Edges connecting consecutive layers
- Node labels (indices)

**Command:**
```bash
hexnet ref -n {n} -r {r} -g structure_matplotlib
```

**Files Generated:**
- `reference/hexnet_n{n}_r{r}_structure.png`

### 2. Activation Structure (`activation`)

Binary matrix visualization showing which connections are active (non-zero) in the weight matrix for a given rotation. This shows the sparsity pattern enforced by the rotation mask.

**Command:**
```bash
hexnet ref -n {n} -r {r} -g activation
```

**Files Generated:**
- `reference/hexnet_n{n}_r{r}_Activation_Structure.png`

### 3. Weight Matrix (`weight`)

Full weight matrix visualization showing actual weight values (typically random initialization for reference graphs).

**Command:**
```bash
hexnet ref -n {n} -r {r} -g weight
```

**Files Generated:**
- `reference/hexnet_n{n}_r{r}_Weight_Matrix.png`

### 4. Multi-Activation Overlay (`multi_activation`)

Overlay visualization showing all 6 rotations' activation patterns simultaneously, each in a different color.

**Command:**
```bash
hexnet ref -n {n} -g multi_activation
```

**Files Generated:**
- `reference/hexnet_n{n}_multi_activation.png`

### 5. Layer Indices (`layer_indices_terminal`)

Text output to console showing how nodes are partitioned into layers for a given rotation. Useful for debugging and understanding the layer structure.

**Command:**
```bash
hexnet ref -n {n} -r {r} -g layer_indices_terminal
```

**Files Generated:**
- None (console output only)

## Complete File Catalog

### For n=2 (7 nodes, 3 layers)

**Structure Graphs:**
- `hexnet_n2_r0_structure.png`
- `hexnet_n2_r1_structure.png`
- `hexnet_n2_r2_structure.png`
- `hexnet_n2_r3_structure.png`
- `hexnet_n2_r4_structure.png`
- `hexnet_n2_r5_structure.png`

**Activation Matrices:**
- `hexnet_n2_r0_Activation_Structure.png`
- `hexnet_n2_r1_Activation_Structure.png`
- `hexnet_n2_r2_Activation_Structure.png`
- `hexnet_n2_r3_Activation_Structure.png`
- `hexnet_n2_r4_Activation_Structure.png`
- `hexnet_n2_r5_Activation_Structure.png`

**Weight Matrices:**
- `hexnet_n2_r0_Weight_Matrix.png`
- `hexnet_n2_r1_Weight_Matrix.png`
- `hexnet_n2_r2_Weight_Matrix.png`
- `hexnet_n2_r3_Weight_Matrix.png`
- `hexnet_n2_r4_Weight_Matrix.png`
- `hexnet_n2_r5_Weight_Matrix.png`

**Multi-Rotation:**
- `hexnet_n2_multi_activation.png`

### For n=3 (19 nodes, 5 layers)

**Structure Graphs:**
- `hexnet_n3_r0_structure.png`
- `hexnet_n3_r1_structure.png`
- `hexnet_n3_r2_structure.png`
- `hexnet_n3_r3_structure.png`
- `hexnet_n3_r4_structure.png`
- `hexnet_n3_r5_structure.png`

**Activation Matrices:**
- `hexnet_n3_r0_Activation_Structure.png`
- `hexnet_n3_r1_Activation_Structure.png`
- `hexnet_n3_r2_Activation_Structure.png`
- `hexnet_n3_r3_Activation_Structure.png`
- `hexnet_n3_r4_Activation_Structure.png`
- `hexnet_n3_r5_Activation_Structure.png`

**Weight Matrices:**
- `hexnet_n3_r0_Weight_Matrix.png`
- `hexnet_n3_r1_Weight_Matrix.png`
- `hexnet_n3_r2_Weight_Matrix.png`
- `hexnet_n3_r3_Weight_Matrix.png`
- `hexnet_n3_r4_Weight_Matrix.png`
- `hexnet_n3_r5_Weight_Matrix.png`

**Multi-Rotation:**
- `hexnet_n3_multi_activation.png`

### For n=4 (28 nodes, 7 layers)

**Structure Graphs:**
- `hexnet_n4_r0_structure.png`
- `hexnet_n4_r1_structure.png`
- `hexnet_n4_r2_structure.png`
- `hexnet_n4_r3_structure.png`
- `hexnet_n4_r4_structure.png`
- `hexnet_n4_r5_structure.png`

**Activation Matrices:**
- `hexnet_n4_r0_Activation_Structure.png`
- `hexnet_n4_r1_Activation_Structure.png`
- `hexnet_n4_r2_Activation_Structure.png`
- `hexnet_n4_r3_Activation_Structure.png`
- `hexnet_n4_r4_Activation_Structure.png`
- `hexnet_n4_r5_Activation_Structure.png`

**Weight Matrices:**
- `hexnet_n4_r0_Weight_Matrix.png`
- `hexnet_n4_r1_Weight_Matrix.png`
- `hexnet_n4_r2_Weight_Matrix.png`
- `hexnet_n4_r3_Weight_Matrix.png`
- `hexnet_n4_r4_Weight_Matrix.png`
- `hexnet_n4_r5_Weight_Matrix.png`

**Multi-Rotation:**
- `hexnet_n4_multi_activation.png`

### For n=5 (40 nodes, 9 layers)

**Structure Graphs:**
- `hexnet_n5_r0_structure.png`
- `hexnet_n5_r1_structure.png`
- `hexnet_n5_r2_structure.png`
- `hexnet_n5_r3_structure.png`
- `hexnet_n5_r4_structure.png`
- `hexnet_n5_r5_structure.png`

**Activation Matrices:**
- `hexnet_n5_r0_Activation_Structure.png`
- `hexnet_n5_r1_Activation_Structure.png`
- `hexnet_n5_r2_Activation_Structure.png`
- `hexnet_n5_r3_Activation_Structure.png`
- `hexnet_n5_r4_Activation_Structure.png`
- `hexnet_n5_r5_Activation_Structure.png`

**Weight Matrices:**
- `hexnet_n5_r0_Weight_Matrix.png`
- `hexnet_n5_r1_Weight_Matrix.png`
- `hexnet_n5_r2_Weight_Matrix.png`
- `hexnet_n5_r3_Weight_Matrix.png`
- `hexnet_n5_r4_Weight_Matrix.png`
- `hexnet_n5_r5_Weight_Matrix.png`

**Multi-Rotation:**
- `hexnet_n5_multi_activation.png`

### For n=6 (54 nodes, 11 layers)

**Structure Graphs:**
- `hexnet_n6_r0_structure.png`
- `hexnet_n6_r1_structure.png`
- `hexnet_n6_r2_structure.png`
- `hexnet_n6_r3_structure.png`
- `hexnet_n6_r4_structure.png`
- `hexnet_n6_r5_structure.png`

**Activation Matrices:**
- `hexnet_n6_r0_Activation_Structure.png`
- `hexnet_n6_r1_Activation_Structure.png`
- `hexnet_n6_r2_Activation_Structure.png`
- `hexnet_n6_r3_Activation_Structure.png`
- `hexnet_n6_r4_Activation_Structure.png`
- `hexnet_n6_r5_Activation_Structure.png`

**Weight Matrices:**
- `hexnet_n6_r0_Weight_Matrix.png`
- `hexnet_n6_r1_Weight_Matrix.png`
- `hexnet_n6_r2_Weight_Matrix.png`
- `hexnet_n6_r3_Weight_Matrix.png`
- `hexnet_n6_r4_Weight_Matrix.png`
- `hexnet_n6_r5_Weight_Matrix.png`

**Multi-Rotation:**
- `hexnet_n6_multi_activation.png`

### For n=7 (70 nodes, 13 layers)

**Structure Graphs:**
- `hexnet_n7_r0_structure.png`
- `hexnet_n7_r1_structure.png`
- `hexnet_n7_r2_structure.png`
- `hexnet_n7_r3_structure.png`
- `hexnet_n7_r4_structure.png`
- `hexnet_n7_r5_structure.png`

**Activation Matrices:**
- `hexnet_n7_r0_Activation_Structure.png`
- `hexnet_n7_r1_Activation_Structure.png`
- `hexnet_n7_r2_Activation_Structure.png`
- `hexnet_n7_r3_Activation_Structure.png`
- `hexnet_n7_r4_Activation_Structure.png`
- `hexnet_n7_r5_Activation_Structure.png`

**Weight Matrices:**
- `hexnet_n7_r0_Weight_Matrix.png`
- `hexnet_n7_r1_Weight_Matrix.png`
- `hexnet_n7_r2_Weight_Matrix.png`
- `hexnet_n7_r3_Weight_Matrix.png`
- `hexnet_n7_r4_Weight_Matrix.png`
- `hexnet_n7_r5_Weight_Matrix.png`

**Multi-Rotation:**
- `hexnet_n7_multi_activation.png`

### For n=8 (88 nodes, 15 layers)

**Structure Graphs:**
- `hexnet_n8_r0_structure.png`
- `hexnet_n8_r1_structure.png`
- `hexnet_n8_r2_structure.png`
- `hexnet_n8_r3_structure.png`
- `hexnet_n8_r4_structure.png`
- `hexnet_n8_r5_structure.png`

**Activation Matrices:**
- `hexnet_n8_r0_Activation_Structure.png`
- `hexnet_n8_r1_Activation_Structure.png`
- `hexnet_n8_r2_Activation_Structure.png`
- `hexnet_n8_r3_Activation_Structure.png`
- `hexnet_n8_r4_Activation_Structure.png`
- `hexnet_n8_r5_Activation_Structure.png`

**Weight Matrices:**
- `hexnet_n8_r0_Weight_Matrix.png`
- `hexnet_n8_r1_Weight_Matrix.png`
- `hexnet_n8_r2_Weight_Matrix.png`
- `hexnet_n8_r3_Weight_Matrix.png`
- `hexnet_n8_r4_Weight_Matrix.png`
- `hexnet_n8_r5_Weight_Matrix.png`

**Multi-Rotation:**
- `hexnet_n8_multi_activation.png`

## Summary Statistics

For each `n`, the following files are generated:

- **6 structure graphs** (one per rotation)
- **6 activation matrices** (one per rotation)
- **6 weight matrices** (one per rotation)
- **1 multi-rotation overlay**

**Total per n:** 19 files (18 rotation-specific + 1 multi-rotation)

**For n ∈ {2, 3, 4, 5, 6, 7, 8}:**
- **Total files:** 7 × 19 = **133 reference files**

Plus terminal output for layer indices (not saved to files).

## Generating All Reference Files

### Using the CLI (Recommended)

The `ref` command now supports a `--all` flag that generates all reference graphs:

```bash
hexnet ref --all
```

or

```bash
python -m src.cli ref --all
```

**Note:** When using `--all`, the `-n` and `-r` arguments are ignored (a warning will be displayed if they are provided).

**From VS Code Tasks:**
- Press `Ctrl+Shift+P` (or `Cmd+Shift+P` on Mac)
- Type "Run Task"
- Select "Generate All Reference Graphs"
