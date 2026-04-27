"""Unit tests for stats_command.py (StatsCommand)."""

import sys
from pathlib import Path

_src = str(Path(__file__).resolve().parent.parent / "src")
if _src in sys.path:
    sys.path.remove(_src)
sys.path.insert(0, _src)
# Do not purge all of `commands.*` here: that evicts `commands.train_command` while other
# test modules may still hold classes bound to the old module, so patches on
# `commands.train_command.RunService` would target the wrong module.

import pytest
from argparse import ArgumentParser, Namespace
from unittest.mock import Mock, patch

from commands.stats_command import StatsCommand

STATS_COMMAND_MODULE = sys.modules["commands.stats_command"]


class TestStatsCommand:
    def setup_method(self):
        self.command = StatsCommand()

    def test_name(self):
        assert self.command.name() == "stats"

    def test_help(self):
        assert len(self.command.help()) > 0
        assert "stat" in self.command.help().lower()

    def test_configure_parser(self):
        parser = ArgumentParser()
        self.command.configure_parser(parser)
        args = parser.parse_args(["runs/my-run"])
        assert args.run_dir == Path("runs/my-run")

    def test_validate_args_run_dir_exists(self):
        run_dir = Path("runs/existing")
        with patch.object(Path, "exists", return_value=True):
            self.command.validate_args(Namespace(run_dir=run_dir))

    def test_validate_args_run_dir_not_exists_raises(self):
        run_dir = Path("runs/missing")
        with patch.object(Path, "exists", return_value=False):
            with pytest.raises(ValueError, match="does not exist"):
                self.command.validate_args(Namespace(run_dir=run_dir))

    @patch.object(STATS_COMMAND_MODULE, "RunService")
    def test_invoke_loads_run_and_shows_stats(self, mock_run_service_class):
        mock_run = Mock()
        mock_run_service_class.return_value = mock_run

        args = Namespace(run_dir=Path("runs/some-run"))
        self.command.invoke(args)

        mock_run_service_class.assert_called_once_with(args)
        mock_run.net.show_stats.assert_called_once()
        mock_run.print_last_training_metrics.assert_called_once()
