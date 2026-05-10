# Vision: Hex Ant Memory Maze

**Purpose:** Capture the system-level idea of a traversable hexagonal maze where each cell is queryable, direction-aware, and potentially trainable by agents moving through the maze.

This is intentionally separate from the benchmark families and the latent attractor work. It is not just a neural-network benchmark. It is an agent/environment simulation built from local HexNet-like computation.

---

## 1. Core Image

Imagine a hexagonal maze.

Each maze cell is itself a small hexagonal computational unit. An ant-like agent stands in a cell, faces a direction, and can query the cell through available, non-walled edges.

If there is a wall in a direction:

```text
the ant cannot query through that side
```

If there is no wall:

```text
the ant can send an input vector into that side
the cell transforms it
an output emerges from the opposite or corresponding side
```

The ant can then use the result to decide:

- where to move
- what to remember
- whether to update the cell
- whether to leave a trace
- whether to modify the maze

Over time, the maze may become a distributed external memory.

---

## 2. Working Names

### Evocative

- The Talking Maze
- Whispering Lattice
- HexLore
- Memory Maze
- The Hex Archive
- Ants in the Oracle Grid

### Technical

- Directional Queryable Hex Lattice
- HexCell Memory Field
- Spatially Addressable HexNet Memory
- Agent-Writable Hexagonal Neural Field
- Local Query Lattice

### Recommended name

> **Hex Ant Memory Maze**

This is clear, memorable, and honest. The technical subtitle can be:

> A Directional Queryable Hex Lattice for Agent-Writable Spatial Memory

---

## 3. Why This Is Its Own Thing

The strange-loop work studies latent recurrence inside a model.

The directional-transfer work studies how learning moves across directions and scales inside a HexNet.

The ant maze studies a larger system:

| Component | Role |
|---|---|
| maze | spatial environment |
| hex cell | local computational unit |
| walls | topological constraint |
| ant | agent / traveler / writer |
| query | local read operation |
| training step | local write operation |
| path | memory trace |
| multiple ants | distributed users of shared memory |

This is closer to:

- artificial life
- differentiable memory
- spatial cognition
- multi-agent systems
- computational art
- reinforcement learning environment
- neural cellular automata, loosely

But the hex geometry and directional querying make it distinct.

---

## 4. Conceptual Model

### 4.1 Maze Grid

The environment is a graph of hexagonal cells.

Each cell has up to six neighbors:

```text
directions = {0, 1, 2, 3, 4, 5}
```

Each edge can be:

| Edge State | Meaning |
|---|---|
| open | ant can move/query |
| wall | ant cannot move/query |
| door | conditionally open |
| fragile wall | can be broken |
| buildable gap | can be closed by ant |
| weighted edge | traversal/query cost varies |

Start with simple open/wall edges.

---

### 4.2 Cell Model

Each cell contains a local function:

```text
y = cell(direction_in, x)
```

Where:

- `x` is an input vector
- `direction_in` is the side being queried
- `y` is output vector
- output may emerge from opposite side or direction-specific output head

Possible implementations:

| Cell Type | Description |
|---|---|
| fixed transform | no learning; easiest start |
| linear map | simple trainable local memory |
| tiny MLP | local nonlinear memory |
| tiny HexNet | full conceptual alignment |
| table / key-value memory | easier interpretability |
| hybrid | queryable model plus stored traces |

Start with fixed transform or linear map. Do not begin with full tiny HexNets unless the simulation scaffolding is already stable.

---

### 4.3 Ant Agent

The ant has:

- position
- facing direction
- optional internal state vector
- movement policy
- query policy
- write policy
- optional energy/budget

Minimal ant state:

```text
position: cell_id
direction: 0..5
memory: vector or none
```

The ant can:

| Action | Meaning |
|---|---|
| move_forward | move through open edge |
| turn_left | rotate direction |
| turn_right | rotate direction |
| query | send vector into current cell/direction |
| write/train | update current cell |
| mark | leave symbolic trace |
| break_wall | modify topology |
| build_wall | modify topology |

Start with:

```text
turn, move, query
```

Add writing later.

---

### 4.4 Query Operation

Basic query:

```text
response = cell.query(direction=facing, input=x)
```

Possible input choices:

| Input | Use |
|---|---|
| zero vector | null prompt |
| ant memory vector | agent-specific state |
| local coordinate vector | position-aware query |
| task vector | goal-directed query |
| random probe | exploration |
| learned query | later-stage policy |

The “null prompt” should be tested carefully. If the cell has no biases or stored state, zero may produce uninteresting output.

---

### 4.5 Write / Training Operation

The ant may update the cell.

Possible write forms:

#### Supervised local write

The ant has target `y_target`:

```text
loss = ||cell.query(d, x) - y_target||
```

Then performs one or more gradient steps.

#### Associative write

The ant stores:

```text
input x should later return output y
```

This makes the cell a local memory surface.

#### Trail write

The ant updates the cell to indicate:

- I was here
- I came from direction d
- this path helped
- this path failed
- this direction leads to goal

#### Value write

The ant writes a scalar value estimate:

```text
this cell/direction is good or bad
```

This resembles reinforcement learning.

---

## 5. Major Research Questions

### 5.1 Can the Maze Become External Memory?

Can an ant remember a path without carrying much internal memory?

Possible mechanism:

- each visited cell is updated
- future queries reveal prior visitation
- useful directions become easier to identify
- dead ends become marked implicitly

Question:

> Can memory live in the environment rather than inside the agent?

---

### 5.2 Can Paths Become Programs?

If each cell transforms a vector and passes it directionally, then a route through the maze becomes a composition of functions.

```text
path = cell_A ∘ cell_B ∘ cell_C ∘ ...
```

Questions:

- Do different paths compute different functions?
- Can a path encode a memory?
- Can the ant discover paths that transform an input into a desired output?
- Can the maze contain multiple computational routes?

---

### 5.3 Can Multiple Ants Communicate Through the Maze?

If ants share cell weights, then one ant’s writes affect another ant’s reads.

Questions:

- Can one ant leave useful traces for another?
- Do ants develop shared route conventions?
- Does multi-agent writing create noise?
- Can agents specialize by territory?
- Can agents sabotage each other accidentally?

---

### 5.4 Can Walls Become Learnable Topology?

If ants can break or build walls, then learning includes graph structure.

Questions:

- Do ants create highways?
- Do they close off unstable or misleading regions?
- Does topology become a memory?
- Are walls used to protect attractors or routes?
- Do multiple ants negotiate or fight over topology?

This is a major escalation. Defer until local querying and writing are understood.

---

### 5.5 Can Local Rules Produce Global Structure?

This is the artificial-life angle.

With only local operations:

- query
- move
- write
- maybe edit wall

Can the system develop:

- trails
- hubs
- territories
- route hierarchies
- stable memory regions
- shared symbolic conventions

---

## 6. Simulation Stages

### Stage 0 — Static Maze, No Cell Computation

Goal:

- implement hex grid
- implement walls
- implement ant movement
- visualize paths

No neural behavior yet.

Deliverables:

- maze renderer
- ant movement
- path trace
- wall map

---

### Stage 1 — Static Queryable Cells

Cells have fixed transforms.

Goal:

- ant can query local cell
- output depends on direction and cell
- no learning yet

Questions:

- Can ant use queries for navigation?
- Are query outputs interpretable?
- Can path composition be measured?

---

### Stage 2 — Trainable Cells, Single Ant

Cells can be updated by ant.

Goal:

- agent writes traces into cells
- future queries reflect past visits

Questions:

- can ant learn path memory externally?
- does repeated traversal stabilize routes?
- does writing cause local overfitting?

---

### Stage 3 — Task-Oriented Single Ant

Give the ant a goal.

Possible tasks:

- find exit
- return to start
- find resource
- map maze
- mark dead ends
- transform input vector by following path

Questions:

- does queryable memory improve task completion?
- does the ant need internal memory?
- how much can be offloaded into the maze?

---

### Stage 4 — Multiple Ants

Multiple agents share the maze.

Questions:

- do agents help or harm each other?
- do they create shared trails?
- do they overwrite one another?
- does population size create phase changes?

---

### Stage 5 — Dynamic Walls

Agents can modify topology.

Questions:

- do they create shortcuts?
- do they isolate memory regions?
- do they build stable infrastructure?
- can topology itself encode learned behavior?

---

### Stage 6 — Full HexNet Cells

Replace simple linear/tiny MLP cells with tiny HexNets.

Questions:

- does directional internal structure improve cell behavior?
- do nested HexNet cells create richer path computation?
- can cells themselves have internal attractors?

This should happen only after earlier stages are stable.

---

## 7. Possible Task Families for the Maze

These are not the same as benchmark families. They are environment tasks.

### 7.1 Navigation Tasks

| Task | Description |
|---|---|
| find exit | reach target cell |
| return home | return to origin after exploration |
| shortest path | improve path length over repeated trials |
| avoid traps | learn unstable or bad cells |
| dynamic maze | adapt after walls change |

---

### 7.2 Memory Tasks

| Task | Description |
|---|---|
| mark visited cells | query reveals previous visitation |
| remember direction home | local query points back toward origin |
| cache goal direction | cells near route store useful heading |
| distributed breadcrumb trail | no single cell stores whole path |

---

### 7.3 Communication Tasks

| Task | Description |
|---|---|
| ant A teaches ant B | A explores, B follows |
| shared map | multiple ants collectively map maze |
| conflicting goals | agents write incompatible traces |
| role specialization | scout ants and worker ants |

---

### 7.4 Computation-by-Path Tasks

| Task | Description |
|---|---|
| path transforms vector | output after route should match target |
| choose route by input | different input vectors use different paths |
| maze as function library | regions encode different transformations |
| path repair | recover computation after wall change |

---

## 8. Metrics

### 8.1 Navigation Metrics

- success rate
- steps to goal
- repeated-trial improvement
- backtracking rate
- coverage efficiency
- shortest-path ratio

---

### 8.2 Memory Metrics

- recall accuracy
- path reconstruction accuracy
- query usefulness
- memory persistence
- memory decay
- overwrite rate

---

### 8.3 Multi-Agent Metrics

- cooperation gain
- interference cost
- trace reuse rate
- shared-route emergence
- territory overlap
- destructive overwrite frequency

---

### 8.4 Topology Metrics

- number of walls changed
- graph connectivity
- average path length
- bottleneck creation
- cluster formation
- route centrality

---

### 8.5 Cell Learning Metrics

- local loss
- number of writes per cell
- weight drift
- query consistency
- direction-specific specialization
- cell saturation / collapse

---

## 9. Visualizations

### 9.1 Maze Map

Show:

- hex grid
- walls
- ant position
- facing direction
- goal
- path history

---

### 9.2 Cell Activity Overlay

Color each cell by:

- number of visits
- number of writes
- query magnitude
- memory confidence
- local loss
- entropy of outputs

---

### 9.3 Directional Edge Overlay

Each cell edge can show:

- wall/open state
- query strength
- learned value
- traffic count
- output norm

This is important because the system is direction-aware.

---

### 9.4 Multi-Ant Trace Map

Each ant has path traces.

Show:

- overlapping paths
- shared highways
- conflict zones
- isolated territories

---

### 9.5 Topology Evolution

Animate wall changes over time.

Useful if ants can build or destroy walls.

---

### 9.6 Query Response Field

For a fixed query vector, show output responses across the maze.

This can reveal whether the maze develops regions with different “meaning.”

---

## 10. Relationship to HexNet Architecture

The maze does not have to use full HexNets at first.

HexNet can appear at three levels:

### 10.1 Cell-Level HexNet

Each maze cell contains a tiny HexNet.

Useful for conceptual purity, but likely expensive and harder to debug.

---

### 10.2 Directional Interface Only

Each cell has six directional inputs/outputs but uses a simple transform internally.

This captures the key behavior without full architecture complexity.

---

### 10.3 Global HexNet-Inspired Environment

The maze itself is the hexagonal computational substrate.

In this interpretation, the entire environment is “HexNet-like,” even if each cell is simple.

Recommended path:

1. build directional interface
2. use simple trainable cells
3. later upgrade cells to tiny HexNets

---

## 11. Relationship to Rotational Dynamics

The ant maze and latent strange-loop ideas rhyme but should remain separate.

| Rotational Dynamics | Ant Maze |
|---|---|
| state moves through latent space | ant moves through physical grid |
| recurrence through one model | traversal through many cells |
| attractor basins | remembered routes / territories |
| loop stability | path stability |
| perturb latent vector | alter maze / wall / cell |
| internal memory | externalized spatial memory |

Later unification:

> The ant’s internal state could itself be a HexLoop, while the maze provides external memory.

Do not start there.

---

## 12. Relationship to Directional Transfer

Directional transfer may show whether one direction’s training affects another direction.

The maze gives that idea a spatial interpretation:

- if a cell is trained from direction 0, does querying from direction 3 change?
- if an ant writes on one wall, does the opposite wall remember?
- if a path is trained eastward, can another ant use it westward?

This could become a local version of the directional transfer matrix.

---

## 13. Failure Modes

| Failure | Meaning |
|---|---|
| ant ignores query outputs | query system not useful for policy |
| cell writes become noise | local learning rule is unstable |
| maze saturates | too much writing destroys distinguishability |
| multi-agent system collapses | overwrite/interference too high |
| walls dominate behavior | topology matters more than cell computation |
| cell computation dominates | maze structure becomes irrelevant |
| no emergent trails | memory mechanism too weak or task too easy |

These are useful diagnostics.

---

## 14. Minimal Prototype Design

### 14.1 Data Structures

```text
Maze:
  cells: dict[cell_id, Cell]
  edges: dict[(cell_id, direction), EdgeState]

Cell:
  model
  query(direction, vector)
  update(direction, input, target)

Ant:
  position
  facing
  memory_vector
  policy
```

---

### 14.2 First Implementation Target

Avoid gradients initially.

Use:

- random fixed vectors per cell/direction
- simple deterministic ant policy
- visible path trace

Then add:

- linear cell model
- local supervised update
- simple task objective

---

### 14.3 First Demo

A clean first demo:

1. Generate a hex maze.
2. Place one ant at start.
3. Ant explores randomly.
4. When it hits a dead end, it writes “bad direction” into the local cell.
5. On future visits, queries help it avoid the dead end.
6. Show path length improving over episodes.

This demonstrates externalized memory without requiring complex neural machinery.

---

## 15. Possible Learning Rules

### 15.1 Dead-End Avoidance

If ant tries direction and fails:

```text
cell.update(direction, query, target = bad)
```

Future query returns a low value for that direction.

---

### 15.2 Breadcrumb Home Vector

When ant moves from cell A to cell B, update B to point back to A.

```text
cell_B.write("home direction", opposite_direction)
```

---

### 15.3 Goal Gradient

When goal is found, back-propagate value along path:

```text
recent cells get higher value for direction leading toward goal
```

This resembles reinforcement learning but can be implemented simply.

---

### 15.4 Associative Memory

Store:

```text
input query vector → output instruction vector
```

This is closer to the “map speaks to you” idea.

---

## 16. Artistic / Conceptual Angle

This system has an unusually strong visual/conceptual identity.

Possible framing:

> A maze that remembers the creatures passing through it.

or:

> A map whose walls become inscriptions.

or:

> A spatial neural memory where navigation is both reading and writing.

This could be a demo, art installation, or interactive paper figure.

The risk is that the concept becomes too poetic before the mechanisms are clear. Keep the first prototype operational and measurable.

---

## 17. Paper-Facing Claim Ladder

| Evidence Level | Claim |
|---|---|
| ant can query cell state | directional queryable maze implemented |
| repeated traversal improves path | local writes support external memory |
| ant with less internal memory performs well | memory is offloaded into environment |
| multiple ants improve each other | shared spatial memory emerges |
| multiple ants degrade each other | overwrite/interference dynamics observed |
| wall editing improves performance | topology becomes part of learning |
| paths compute transformations | maze acts as spatial computational graph |

---

## 18. Summary

The Hex Ant Memory Maze is a system-level exploration.

Its core idea:

> A traversable hexagonal environment where agents read from and write to local direction-aware computational cells, allowing memory to emerge in the map itself.

This should remain separate from the first benchmark paper, but it is a strong long-term demo and research direction. It gives the HexNet project an embodied, visual, and interactive extension without forcing it into the initial empirical matrix.
