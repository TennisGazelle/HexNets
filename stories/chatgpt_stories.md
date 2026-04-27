Recommended label scheme

Since GitHub Issues does not have true epics by default, I’d do this with labels:

epic:core-correctness
epic:dataset-benchmarks
epic:experiment-orchestration
epic:streamlit
epic:transfer-lesioning
epic:paper-support

And support labels:

cli
streamlit
tests
docs
research
paper
good-first-deep-work
Ordered issue backlog

These are written as actionable stories with checklists, and each one includes a Streamlit page or enhancement where it makes sense.

Epic: Core Correctness
Story 1 — Audit and fix regression metric calculations

Labels: epic:core-correctness, tests, research, paper

Why
You are already tracking loss, accuracy, r_squared, and adjusted_r_squared, but the paper cannot lean on them until they are validated.

Checklist

 Review src/networks/metrics.py and document the intended meaning of each metric
 Verify accuracy proxy behavior for regression and decide whether to rename it in code and UI
 Verify r_squared and adjusted_r_squared formulas against known toy data
 Add unit tests for perfect fit, poor fit, and low-sample edge cases
 Add a short metrics interpretation section to docs
 Update any labels in Streamlit that are misleading for regression

Streamlit

 Add a Metrics Explainer expander or page showing formulas, caveats, and example values

Definition of done

Metric tests pass
Docs reflect final definitions
Streamlit labels match the final terminology
Story 2 — Fix CLI / docs / code naming inconsistencies

Labels: epic:core-correctness, cli, docs

Why
There are a few naming drifts that will create friction later, such as the CLI dataset type linear while the dataset class registers as linear_scale, and stats_conmand.py having a typo in the filename.

Checklist

 Normalize dataset naming between CLI options, dataset display names, docs, and manifests
 Rename stats_conmand.py to stats_command.py
 Confirm import paths remain stable after rename
 Update any docs that reference old names
 Add one smoke test covering the renamed stats command

Streamlit

 Update dataset names in dropdowns and captions so they match CLI/docs exactly

Definition of done

No typoed command module names remain
Dataset names are consistent across CLI, docs, saved configs, and Streamlit
Story 3 — Add run manifest fields needed for paper traceability

Labels: epic:core-correctness, paper, cli

Why
Run persistence exists already, but it is still fairly minimal for paper traceability.

Checklist

 Add git commit SHA to run manifest
 Add random seed to persisted config/manifest
 Add dataset metadata block to persisted config
 Add model parameter count to manifest
 Add optional freeform note / tag field for experiments
 Preserve backward compatibility when loading older runs

Streamlit

 Add a Run Metadata panel for any loaded run

Definition of done

New runs contain paper-useful provenance
Old runs still load cleanly
Epic: Dataset Benchmarks
Story 4 — Promote dataset registry to first-class CLI support

Labels: epic:dataset-benchmarks, cli, tests

Why
You already have more dataset structure than the CLI exposes. Right now the command helper hardcodes only identity and linear.

Checklist

 Refactor dataset selection to use the dataset registry instead of hardcoded branching
 Expose diagonal_scale through CLI
 Ensure saved run config stores the exact dataset display name
 Add tests for selecting each registered dataset
 Document how to add new datasets

Streamlit

 Replace the fixed training dataset behavior with a dataset selector using the registry

Definition of done

Adding a dataset no longer requires multiple manual switch statements
CLI and Streamlit both discover the same dataset list
Story 5 — Add noisy synthetic regression datasets

Labels: epic:dataset-benchmarks, research, cli, streamlit

Why
Your living doc keeps pointing toward noisy / fuzzy regression as a valuable proving ground.

Checklist

 Add configurable Gaussian noise to inputs
 Add configurable Gaussian noise to targets
 Support combined input+target noise
 Save noise parameters into run config
 Add unit tests confirming noise application and shape stability
 Update docs with mathematical form and examples

Streamlit

 Add a Dataset Explorer page showing clean vs noisy data samples
 Add sliders for noise level and a small preview chart/table

Definition of done

Noise can be turned on from CLI and Streamlit
Run metadata makes noisy runs distinguishable from clean runs
Story 6 — Add two more same-dimension benchmark datasets

Labels: epic:dataset-benchmarks, research, paper

Why
A first paper will be stronger if it is not basically “identity + linear scaling only.”

Checklist

 Add one affine or permutation-based same-dimension dataset
 Add one nonlinear but controlled same-dimension dataset
 Add tests for output shapes and expected transforms
 Document why each dataset is useful as a benchmark
 Add sample generation examples to docs

Streamlit

 Extend Dataset Explorer to preview all available datasets side by side

Definition of done

Benchmark suite includes at least 5 named datasets
Each dataset is documented and selectable from CLI/Streamlit
Epic: Experiment Orchestration
Story 7 — Build a sweep command for batch experiments

Labels: epic:experiment-orchestration, cli, research, paper

Why
You already have single-run training. What is missing is a first-class mechanism for controlled sweeps.

Checklist

 Add a new CLI command for parameter sweeps
 Support sweeping over dataset, activation, loss, model, n, rotation, and learning rate schedule
 Skip or resume already-completed runs
 Emit a summary CSV/JSON of completed sweep outputs
 Add tests for sweep plan generation
 Document usage examples

Streamlit

 Add a Sweep Planner page that previews the run matrix before execution
 Show total planned runs and estimated artifact count

Definition of done

You can define a repeatable sweep without shell scripting it by hand
Sweep outputs aggregate into a summary artifact
Story 8 — Add non-constant learning-rate schedules

Labels: epic:experiment-orchestration, cli, tests, streamlit

Why
The architecture for learning-rate plugins exists, but only constant is actually implemented.

Checklist

 Implement exponential decay schedule
 Implement one additional schedule, likely rolling decay or step decay
 Add tests for rate_at_iteration() behavior
 Ensure schedules serialize cleanly into run config
 Update reference generation so learning-rate figures include new schedules

Streamlit

 Add a Learning Rate Schedules page using the existing reference-figure concept
 Add interactive plots comparing schedules

Definition of done

At least 3 schedules are selectable from CLI and viewable in Streamlit
Story 9 — Add aggregate run table and comparison utilities

Labels: epic:experiment-orchestration, paper, streamlit

Why
You have run folders, but not yet a strong compare-and-rank layer.

Checklist

 Create a utility that scans runs/ and builds a consolidated dataframe/table
 Include final and best metrics per run
 Include filter columns for dataset, activation, loss, model, n, rotation, seed
 Export markdown and CSV summaries
 Add tests for parsing runs with missing or older fields

Streamlit

 Add a Run Browser page with filters and sortable tables
 Let a user click into a run to see metadata + metrics + plots

Definition of done

Completed runs can be compared without manually opening JSON files
Epic: Streamlit
Story 10 — Upgrade Streamlit from visualizer to experiment workbench

Labels: epic:streamlit, streamlit, paper

Why
The current app is useful, but still narrow: two tabs, one quick training button, and reference image loading.

Checklist

 Split app into multi-page navigation
 Keep current Network Explorer page
 Keep current Rotation Comparison page
 Add placeholders/routes for Dataset Explorer, Run Browser, Sweep Planner, Lesion Lab
 Refactor shared UI helpers out of streamlit_app.py
 Document page layout in docs/streamlit_app.md

Definition of done

App has a clear research-workbench structure rather than a single-file demo feel
Story 11 — Add a Run Browser page

Labels: epic:streamlit, streamlit, paper

Checklist

 List runs from runs/
 Filter by model, dataset, n, rotation, activation, loss
 Show run metadata and persisted config
 Show training curves for selected run
 Gracefully handle missing legacy fields

Definition of done

A completed run can be inspected entirely from Streamlit
Story 12 — Add a Dataset Explorer page

Labels: epic:streamlit, streamlit, research

Checklist

 Select dataset type
 Adjust sample count and dimension
 Adjust noise settings where applicable
 Show a sample table and summary stats
 Add a short description of what transformation the dataset represents

Definition of done

You can visually inspect benchmark datasets before training on them
Story 13 — Add a Benchmark Comparison page

Labels: epic:streamlit, streamlit, paper

Checklist

 Select multiple runs for comparison
 Overlay training curves
 Compare final metrics in one table
 Allow HexNet vs MLP comparison
 Export a comparison table for paper drafting

Definition of done

You can compare candidate paper figures from the UI instead of manually splicing plots
Epic: Transfer & Lesioning

This is where the real paper-value starts.

Story 14 — Implement MLP baseline parity runs

Labels: epic:transfer-lesioning, research, paper, cli

Why
The repo already has MLPNetwork, but it needs to become a proper baseline in the experiment workflow, not just a side model.

Checklist

 Make MLP baseline selectable in sweep command
 Record model parameter count for HexNet and MLP
 Document how hidden dims are chosen
 Add comparison tests ensuring MLP and HexNet runs serialize in a common format
 Add example benchmark doc section for HexNet vs MLP

Streamlit

 Add MLP vs HexNet toggle to Benchmark Comparison page

Definition of done

HexNet claims can always be contextualized against an MLP baseline
Story 15 — Add sequential-task training workflow

Labels: epic:transfer-lesioning, research, cli

Why
Your living doc’s strongest near-term research path is sequential training and retention.

Checklist

 Add a workflow for training Task A then Task B in one run
 Persist pre-task-B and post-task-B evaluation metrics
 Support selecting dataset A and dataset B independently
 Save retention metrics into run outputs
 Add tests for sequential-run artifact structure

Streamlit

 Add a Sequential Task Lab page showing before/after metrics and forgetting curves

Definition of done

You can measure retention degradation across two tasks using a single formal workflow
Story 16 — Add cross-direction training protocol support

Labels: epic:transfer-lesioning, research, cli

Checklist

 Implement protocol A: train one direction then another
 Implement protocol B: alternate directions every k epochs
 Persist the chosen protocol in run config
 Add tests for protocol selection and metadata
 Add docs describing how each protocol maps to your research question

Streamlit

 Add protocol selector and explanation to Sequential Task Lab

Definition of done

Directional transfer experiments are a first-class feature instead of custom scripts
Story 17 — Add lesion framework with uniform baseline

Labels: epic:transfer-lesioning, research, paper, cli

Why
This is the path to your best claim: “geometry matters, not just sparsity.”

Checklist

 Add a lesion abstraction layer
 Implement uniform random edge deletion baseline
 Support configuring lesion fraction and seed
 Persist lesion metadata to run config/manifest
 Add tests confirming lesions modify only intended weights

Streamlit

 Add a Lesion Lab page with a lesion-type selector and lesion preview summary

Definition of done

Uniform lesioning can be run, saved, compared, and reproduced
Story 18 — Add directional lesion types

Labels: epic:transfer-lesioning, research, paper, streamlit

Checklist

 Implement single-direction lesion
 Implement opposing-direction lesion
 Implement sector lesion
 Implement rotating-direction lesion
 Add validation and metadata for all lesion modes
 Add tests confirming targeted scope of each lesion type

Streamlit

 In Lesion Lab, visualize which structural region/direction is being lesioned
 Show side-by-side lesion mask summaries

Definition of done

Directional lesions are operational and inspectable
Story 19 — Add no-lesion vs uniform vs directional retention benchmark

Labels: epic:transfer-lesioning, research, paper

Checklist

 Run sequential-task workflow without lesion
 Run sequential-task workflow with uniform lesion
 Run sequential-task workflow with directional lesion
 Export one summary table comparing retention degradation
 Export one plot intended for paper use
 Add doc section describing interpretation boundaries

Streamlit

 Add a one-click comparison view in Lesion Lab

Definition of done

You have a reproducible benchmark that can support the geometry-vs-sparsity argument

This is the most important medium-term story in the whole backlog.

Epic: Paper Support
Story 20 — Add paper-figure export and traceability mapping

Labels: epic:paper-support, paper, docs

Checklist

 Add a script or CLI command that exports selected run plots into a paper-figures directory
 Generate a manifest mapping figure file → run ID → config
 Add markdown snippets for figure captions / notes
 Document how to reproduce each exported figure

Streamlit

 Add an “Export for paper” action from Run Browser / Benchmark Comparison page

Definition of done

Every figure you use in the paper has a reproducible provenance trail
Story 21 — Add literature-to-experiment mapping doc

Labels: epic:paper-support, paper, docs

Checklist

 Create a markdown table mapping cited papers to project components
 Mark each citation as framing, metrics, lesion precedent, continual learning, or interpretation
 Link each citation to the relevant experiment or future work item
 Keep it next to the paper workflow docs

Streamlit

 None required

Definition of done

The paper outline and implementation backlog stop drifting apart





Recommended order

If you want the practical sequence, I would do:

Phase 1
Story 1 — Audit and fix regression metric calculations
Story 2 — Fix CLI / docs / code naming inconsistencies
Story 3 — Add run manifest fields needed for paper traceability
Story 4 — Promote dataset registry to first-class CLI support
Story 10 — Upgrade Streamlit from visualizer to experiment workbench
Phase 2
Story 5 — Add noisy synthetic regression datasets
Story 6 — Add two more same-dimension benchmark datasets
Story 8 — Add non-constant learning-rate schedules
Story 7 — Build a sweep command for batch experiments
Story 9 — Add aggregate run table and comparison utilities
Story 11 — Add a Run Browser page
Story 12 — Add a Dataset Explorer page
Story 13 — Add a Benchmark Comparison page
Phase 3
Story 14 — Implement MLP baseline parity runs
Story 15 — Add sequential-task training workflow
Story 16 — Add cross-direction training protocol support
Story 17 — Add lesion framework with uniform baseline
Story 18 — Add directional lesion types
Story 19 — Add no-lesion vs uniform vs directional retention benchmark
Phase 4
Story 20 — Add paper-figure export and traceability mapping
Story 21 — Add literature-to-experiment mapping doc
My blunt read on priorities

The highest-value near-term combo is:

metric audit
dataset expansion
sweep runner
run browser / comparison UI
sequential-task + lesion baseline

That gets you from “promising framework” to “I can start generating real paper evidence.”

The lower-value work right now would be:

more cosmetic reference visualizations
exotic psychological framing features
recurrent / DID / personality interpretations

Not because they are bad, but because they are downstream of experimental proof.

One implementation detail I would strongly suggest

For each issue, use this body template:

Goal

One sentence.

Why

Tie it to either:

correctness
benchmark coverage
reproducibility
paper evidence
Checklist

Concrete boxes only.

Streamlit surface

State the page or enhancement explicitly, or say “none”.

Acceptance criteria

Observable outcome, not intention.

If you want, next I’ll turn these into copy-paste GitHub issue bodies for the first 8 stories, already formatted with checkboxes and label suggestions.