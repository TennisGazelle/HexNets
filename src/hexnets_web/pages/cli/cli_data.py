"""
CLI builder tree aggregation (no Streamlit imports).

Same registration order as src/cli.py subparsers.
"""

from __future__ import annotations

from commands.adhoc_command import AdhocCommand
from commands.reference_command import ReferenceCommand
from commands.stats_command import StatsCommand
from commands.maze_command import MazeCommand
from commands.train_command import TrainCommand
from hexnets_web.cli_types import CliNode

_COMMAND_CLASSES = (ReferenceCommand, AdhocCommand, TrainCommand, StatsCommand, MazeCommand)


def build_cli_root() -> CliNode:
    return CliNode(
        name="hexnet",
        help="Hexagonal Neural Network CLI tool",
        children=tuple(c.get_cli_node() for c in _COMMAND_CLASSES),
    )


CLI_ROOT: CliNode = build_cli_root()
