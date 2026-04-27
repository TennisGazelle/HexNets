"""Unit tests for train_command.py"""

import sys
from pathlib import Path

_src = str(Path(__file__).resolve().parent.parent / "src")
if _src in sys.path:
    sys.path.remove(_src)
sys.path.insert(0, _src)
for _k in list(sys.modules):
    if _k == "commands" or (_k.startswith("commands.") and "test_" not in _k):
        del sys.modules[_k]

import pytest
from argparse import ArgumentParser, Namespace
from unittest.mock import Mock, patch

from commands.train_command import TrainCommand


def _valid_train_args(**overrides):
    defaults = dict(
        n=2,
        rotation=0,
        activation="sigmoid",
        loss="mean_squared_error",
        learning_rate="constant",
        epochs=2,
        pause=0.0,
        type="identity",
        dataset_size=20,
        seed=42,
        run_dir=None,
        run_name=None,
        dry_run=False,
    )
    defaults.update(overrides)
    return Namespace(**defaults)


class TestTrainCommand:
    def setup_method(self):
        self.command = TrainCommand()

    def test_name(self):
        assert self.command.name() == "train"

    def test_help(self):
        assert len(self.command.help()) > 0
        assert "train" in self.command.help().lower()

    def test_configure_parser(self):
        parser = ArgumentParser()
        self.command.configure_parser(parser)
        args = parser.parse_args(
            [
                "-n", "3", "-r", "0", "-e", "10", "-t", "linear_scale",
                "-rn", "my-run", "--dry-run",
            ]
        )
        assert args.n == 3
        assert args.run_name == "my-run"
        assert args.dry_run is True
        args2 = parser.parse_args(["-rd", "runs/some-run"])
        assert args2.run_dir is not None
        assert str(args2.run_dir) == "runs/some-run"

    def test_validate_args_valid_no_run_name(self):
        self.command.validate_args(_valid_train_args())

    def test_validate_args_valid_with_run_name(self):
        with patch("commands.train_command.RunService") as mock_rs:
            mock_runs_dir = Mock()
            mock_runs_dir.__truediv__ = Mock(return_value=Mock(exists=Mock(return_value=False)))
            mock_rs.runs_dir = mock_runs_dir
            self.command.validate_args(_valid_train_args(run_name="new-run"))

    def test_validate_args_run_name_and_run_dir_raises(self):
        with pytest.raises(ValueError, match="run_name and have a run_dir"):
            self.command.validate_args(
                _valid_train_args(run_name="x", run_dir=Path("runs/x"))
            )

    def test_validate_args_run_name_already_exists_raises(self):
        with patch("commands.train_command.RunService") as mock_rs:
            mock_runs_dir = Mock()
            mock_runs_dir.__truediv__ = Mock(return_value=Mock(exists=Mock(return_value=True)))
            mock_rs.runs_dir = mock_runs_dir
            with pytest.raises(ValueError, match="already exists"):
                self.command.validate_args(_valid_train_args(run_name="taken"))

    def test_validate_args_run_dir_not_exists_raises(self):
        args = _valid_train_args(run_dir=Path("/nonexistent/path/for/test"))
        with pytest.raises(ValueError, match="does not exist"):
            self.command.validate_args(args)

    def test_validate_args_hex_invalid_n(self):
        with pytest.raises(ValueError, match="at least 2"):
            self.command.validate_args(_valid_train_args(n=1))

    @patch("commands.train_command.get_dataset")
    @patch("commands.train_command.RunService")
    def test_invoke_identity_calls_run_service(
        self, mock_run_service_class, mock_get_dataset
    ):
        mock_run = Mock()
        mock_run.get_figures_path.return_value = Path("figures")
        mock_run.get_network_weights_path.return_value = Path("runs/x/weights")
        mock_run.net = Mock()
        mock_run_service_class.return_value = mock_run
        mock_get_dataset.return_value = Mock()

        args = _valid_train_args(type="identity")
        self.command.invoke(args)

        mock_run_service_class.assert_called_once_with(args)
        mock_run.output_run_files.assert_called()
        mock_run.net.train_animated.assert_called()

    @patch("commands.train_command.get_dataset")
    @patch("commands.train_command.RunService")
    def test_invoke_dry_run_skips_output(self, mock_run_service_class, mock_get_dataset):
        mock_run = Mock()
        mock_run.get_figures_path.return_value = Path("figures")
        mock_run.net = Mock()
        mock_run_service_class.return_value = mock_run
        mock_get_dataset.return_value = Mock()

        args = _valid_train_args(dry_run=True)
        self.command.invoke(args)

        mock_run.output_run_files.assert_not_called()

    def test_invoke_invalid_type_raises(self):
        args = _valid_train_args(type="invalid")
        with pytest.raises(ValueError, match="Invalid dataset type"):
            self.command.invoke(args)
