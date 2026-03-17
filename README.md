# HexNets
Hexagonal Neural Network Implementations

## Nomenclature

Reference Graphs - figures that do not belong to a run, but is instead created given the structure of the graph.  For examples, while weights may vary, all hex graphs of n=4 will have the same activation structure and activation weight matrix.

Training Graph - The graph showing the progression of loss, accuracy, r^2, and adjusted r^2 of a run.

Run - A run consists of the implementation and execution of a model.

## CLI Interface

When installing, a figures/ and a runs/ directory will be created.
All runs will be saved by their name (if not provided, one will be generated with a timestamp and a uuid segment), underneath the run directory.
The figures directory will hold all figures related to reference graphs ()