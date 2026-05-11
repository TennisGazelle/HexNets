# HexNet Research Vision Roadmap

**Purpose:** This document organizes the major research tracks around Hexagonal Neural Networks after the baseline benchmark families are assumed to exist. It separates the project into empirical benchmarks, architecture-specific behavior studies, dynamical-systems investigations, and larger speculative systems.

The benchmark families remain the canonical reference for controlled synthetic task comparison. These vision files are intended to capture the larger research directions that are not yet cleanly reducible to a dataset × model × activation × loss experiment matrix.

---

## 1. Project Layering

HexNet work is now naturally separating into four layers:

| Layer | Main Question | Primary Document |
|---|---|---|
| Benchmark matrix | Does HexNet behave differently from MLPs on controlled task families? | `benchmark-families.md` |
| Directional behavior | Does learning in one direction, rotation, or scale transfer elsewhere? | `02-directional-transfer-and-nested-cores.md` |
| Dynamical behavior | What trajectories, loops, attractors, and stable regimes emerge under repeated directional application? | `01-rotational-dynamics-and-attractors.md` |
| Embodied / spatial systems | What happens when HexNet-like cells become a navigable memory environment? | `03-hex-ant-memory-maze.md` |

The first layer is about **performance under controlled tasks**. The later layers are about **behavior, structure, memory, recurrence, and system-level emergence**.

---

## 2. Current Maturity Map

| Track | Maturity | Near-Term Use | Long-Term Use |
|---|---:|---|---|
| Benchmark families | High | Main empirical baseline | Paper results section |
| Directional transfer / nested cores | Medium | HexNet-specific behavioral extension | Possible new benchmark family |
| Rotational dynamics / attractors | Medium-low | Exploratory analysis | Theoretical / dynamical-systems paper thread |
| Hex ant memory maze | Low but distinctive | Simulation prototype | Demo, visualization, art/research hybrid, agent-memory system |

---

## 3. Relationship to the Existing Benchmark Families

The benchmark matrix answers questions like:

- Can HexNet match MLP on trivial maps?
- Does HexNet separate from MLP on structured linear, geometric, nonlinear, noisy, or discontinuous tasks?
- Are differences family-dependent?
- Does activation or loss dominate architecture?
- Does robustness emerge under noise?

The new vision tracks answer different questions:

- What does a trained direction imply about an untrained direction?
- Can HexNet contain reusable trained substructures?
- Can repeated application of HexNet generate stable loops or attractors?
- Can an external spatial environment made of queryable HexNet cells behave like memory?
- Can multiple agents write into and read from such an environment?

These should not be forced into the benchmark matrix too early. They need their own vocabulary first.

---

## 4. Research Tracks

### Track A — Controlled Benchmarks

**File:** `benchmark-families.md`

**Role:** Empirical foundation.

This is the least speculative part of the project. It should remain disciplined and restrained:

- HexNet vs MLP
- synthetic task families
- sweep only a small set of variables at a time
- use clean baselines
- avoid overclaiming
- treat failure cases as useful evidence

**Best paper role:** Initial results, calibration, and architectural comparison.

---

### Track B — Directional Transfer and Nested Cores

**File:** `02-directional-transfer-and-nested-cores.md`

**Role:** First HexNet-specific behavioral extension. 

This track asks whether geometry induces useful coupling across directions and scales.

Core idea examples:

- Train rotations `0, 2, 4`; test `1, 3, 5`.
- Train one direction; test all others.
- Train opposite direction pairs.
- Replace the inside of a larger HexNet with a trained smaller HexNet.
- Sweep the ratio of trained inner radius to full radius.
- Freeze vs fine-tune the inserted core.

**Why it matters:**  
This is where HexNet starts to look unlike merely “an oddly shaped MLP.” If directional training creates predictable transfer, interference, modularity, or scale effects, then the hexagonal structure has behavioral consequences beyond raw performance.

**Best paper role:** Follow-up section after the benchmark matrix, possibly a separate “architectural behavior” section.

---

### Track C — Rotational Dynamics and Attractors

**File:** `01-rotational-dynamics-and-attractors.md`

**Role:** Dynamical-systems research path.

This track treats HexNet as something that can be iterated:

```text
x_{t+1} = R_k(f(x_t))
```

Where:

- `x_t` is a latent vector at step `t`
- `f` is a directional HexNet pass
- `R_k` is a rotation or directional projection schedule
- the sequence of rotations defines a trajectory

This track includes:

- strange-loop style recurrence
- fixed points
- limit cycles
- attractor splitting
- persona-like regimes
- orbit stability
- perturbation recovery
- loop interference
- directional lesioning

**Why it matters:**  
This is the most theoretically rich track. It connects HexNet to recurrent dynamics, attractor networks, stability analysis, and structured latent motion.

**Best paper role:** A future paper or later theoretical section once the benchmark foundation is credible.

---

### Track D — Hex Ant Memory Maze

**File:** `03-hex-ant-memory-maze.md`

**Role:** Spatial simulation / embodied memory system.

This track imagines a hex-grid maze where each cell contains a small queryable model. An ant-like agent moves through the grid, queries accessible directions, receives outputs, and may update the cell’s local model.

The maze becomes:

- a traversable environment
- a distributed memory
- a set of directional neural cells
- a shared substrate for multiple agents
- potentially a topology that agents can modify

**Why it matters:**  
This is not simply a benchmark or a neural architecture. It is a system. It may become a demo, simulation, artificial-life environment, or artistic-computational artifact.

**Best paper role:** Probably not the first paper. It is more suitable as a later demo, visualization, or exploratory system.

---

## 5. Suggested Development Order

### Phase 1 — Stabilize the Benchmark Foundation

Focus:

- finish benchmark family implementations
- verify CLI consistency
- produce repeatable plots
- collect loss curves and seed variance
- identify where HexNet behaves similarly or differently from MLP

Output:

- baseline table
- convergence curves
- selected noise curves
- one restrained claims section

---

### Phase 2 — Add Directional Transfer

Focus:

- train one direction, evaluate all directions
- train alternating rotations, evaluate complement
- compute transfer/interference matrices
- test whether untrained rotations drift, improve, or degrade

Output:

- rotation-to-rotation transfer matrix
- direction-specific performance plots
- trained-vs-untrained rotation comparisons

This is the cleanest first extension because it directly relies on HexNet geometry.

---

### Phase 3 — Add Nested-Core Experiments

Focus:

- train smaller HexNet
- insert into larger HexNet interior
- freeze vs fine-tune
- sweep core radius ratios
- compare against random initialization

Output:

- convergence speed comparison
- final loss comparison
- core preservation metrics
- radial gradient magnitude plots

This tests whether HexNet has a useful recursive structure.

---

### Phase 4 — Explore Attractor Dynamics

Focus:

- iterate trained and untrained HexNets
- detect fixed points and cycles
- compare rotation schedules
- perturb trajectories
- test lesion effects

Output:

- latent trajectory plots
- recurrence-distance heatmaps
- cycle stability scores
- attractor basin maps

This should come after the model and instrumentation are mature enough.

---

### Phase 5 — Prototype Hex Ant Maze

Focus:

- start without learning
- build a traversable hex grid
- add direction-aware cell queries
- add local writing later
- add multiple ants only after the single-agent case is understandable

Output:

- interactive simulation
- path traces
- cell-state heatmaps
- multi-agent interference/collaboration experiments

This is a later system-level exploration.

---

## 6. Naming Conventions

Recommended names:

| Concept | Working Name | More Technical Name |
|---|---|---|
| Strange-loop dynamics | HexLoops | Rotational Latent Orbit Dynamics |
| Attractor splitting | Attractor Splitting | Directional Attractor Branching |
| Direction transfer | Rotational Transfer | Directional Transfer Matrix |
| Nested trained core | HexCore Transplant | Nested-Core Initialization |
| Ant maze | Hex Ant Memory Maze | Directional Queryable Hex Lattice |

The working names can be used in notes and demos. The technical names can be used in papers.

---

## 7. What Not To Collapse Together Too Early

Avoid merging these prematurely:

| Do Not Merge Yet | Reason |
|---|---|
| Benchmark families + ant maze | One is controlled empirical comparison; the other is a full agent/environment system. |
| Ant maze + strange loops | They rhyme conceptually, but one is spatial traversal and the other is latent recurrence. |
| Directional transfer + generic benchmark matrix | Directional transfer needs HexNet-specific instrumentation and would clutter the baseline matrix. |
| Persona claims + early results | “Persona” should remain tentative unless backed by stable attractor or behavioral separation evidence. |

---

## 8. Paper Positioning

A restrained first-paper structure could look like:

1. Motivation and architecture
2. Benchmark families
3. Results on controlled task families
4. Directional transfer as a HexNet-specific behavior
5. Limitations
6. Future work:
   - nested cores
   - attractor dynamics
   - ant memory maze

The first paper should probably not lead with strange loops, personas, or ant mazes. Those are compelling, but they need the empirical base first.

---

## 9. North Star

A concise unifying statement:

> Hexagonal Neural Networks are not only an alternative feedforward topology; they are a structured substrate for studying directional computation, geometric transfer, recursive modularity, and latent dynamical behavior.

The benchmark families establish whether the substrate is empirically sane. The vision tracks explore what the substrate can become.
