"""Unit tests for reference_command.py"""

import sys
from pathlib import Path

# Ensure src is first on path (pytest prepends the test dir, so we must put src first for imports)
_src = Path(__file__).resolve().parent.parent.parent / "src"
_src_str = str(_src)
if _src_str in sys.path:
    sys.path.remove(_src_str)
sys.path.insert(0, _src_str)
import pytest
from argparse import Namespace
from unittest.mock import Mock, patch, MagicMock, call
import numpy as np

# Clear only the app's commands package (not test module names); tests/conftest can load before path is fixed
for key in list(sys.modules):
    if key == "commands" or (key.startswith("commands.") and "test_" not in key):
        del sys.modules[key]
from commands.reference_command import ReferenceCommand


class TestReferenceCommand:
    """Test cases for ReferenceCommand"""

    def setup_method(self):
        """Set up test fixtures"""
        self.command = ReferenceCommand()

    def test_name(self):
        """Test that the command name is correct"""
        assert self.command.name() == "ref"

    def test_help(self):
        """Test that help text is provided"""
        assert len(self.command.help()) > 0
        assert "reference" in self.command.help().lower()

    def test_configure_parser(self):
        """Test that parser is configured with expected arguments"""
        from argparse import ArgumentParser
        parser = ArgumentParser()
        self.command.configure_parser(parser)
        
        # Check that parser has the expected arguments
        args = parser.parse_args(["--graph", "activation", "--dry-run"])
        assert args.graph == "activation"
        assert args.dry_run is True

    def test_validate_args_no_params(self):
        """Test validation fails when no parameters are specified"""
        args = Namespace(
            model="hex",
            n=None,
            rotation=None,
            graph=None,
            generate_all=False,
            activation="sigmoid",
            loss="mean_squared_error",
        )
        
        with pytest.raises(ValueError, match="No parameters specified"):
            self.command.validate_args(args)

    def test_validate_args_with_all_flag(self):
        """Test validation with --all flag"""
        args = Namespace(
            model="hex",
            n=None,
            rotation=None,
            graph=None,
            generate_all=True,
            activation="sigmoid",
            loss="mean_squared_error",
        )
        
        # Should not raise an error
        self.command.validate_args(args)

    def test_validate_args_all_with_mlp(self):
        """Test that --all flag cannot be used with mlp model"""
        args = Namespace(
            model="mlp",
            n=None,
            rotation=None,
            graph=None,
            generate_all=True,
            activation="sigmoid",
            loss="mean_squared_error",
        )
        
        with pytest.raises(ValueError, match="--all flag cannot be used with mlp"):
            self.command.validate_args(args)

    def test_validate_args_none_model(self):
        """Test validation passes for model='none'"""
        args = Namespace(
            model="none",
            n=None,
            rotation=None,
            graph=None,
            generate_all=False,
        )
        
        # Should not raise an error
        self.command.validate_args(args)

    def test_validate_args_invalid_n(self):
        """Test validation fails with invalid n value"""
        args = Namespace(
            model="hex",
            n=1,  # Too small
            rotation=None,
            graph=None,
            generate_all=False,
            activation="sigmoid",
            loss="mean_squared_error",
        )
        
        with pytest.raises(ValueError):
            self.command.validate_args(args)

    def test_validate_args_invalid_rotation(self):
        """Test validation fails with invalid rotation value"""
        args = Namespace(
            model="hex",
            n=3,
            rotation=6,  # Too large
            graph=None,
            generate_all=False,
            activation="sigmoid",
            loss="mean_squared_error",
        )
        
        with pytest.raises(ValueError):
            self.command.validate_args(args)

    def test_determine_iteration_ranges_all(self):
        """Test iteration ranges when --all is specified"""
        args = Namespace(
            generate_all=True,
            n=None,
            rotation=None,
            graph=None,
        )
        
        n_range, r_range, graph_types = self.command._determine_iteration_ranges(args)
        
        assert list(n_range) == list(range(2, 9))
        assert list(r_range) == list(range(6))
        assert len(graph_types) == 6  # All graph types

    def test_determine_iteration_ranges_specific_n(self):
        """Test iteration ranges when n is specified"""
        args = Namespace(
            generate_all=False,
            n=3,
            rotation=None,
            graph=None,
        )
        
        n_range, r_range, graph_types = self.command._determine_iteration_ranges(args)
        
        assert list(n_range) == [3]
        assert list(r_range) == list(range(6))
        assert len(graph_types) == 6

    def test_determine_iteration_ranges_specific_r(self):
        """Test iteration ranges when r is specified"""
        args = Namespace(
            generate_all=False,
            n=None,
            rotation=2,
            graph=None,
        )
        
        n_range, r_range, graph_types = self.command._determine_iteration_ranges(args)
        
        assert list(n_range) == list(range(2, 9))
        assert list(r_range) == [2]
        assert len(graph_types) == 6

    def test_determine_iteration_ranges_specific_graph(self):
        """Test iteration ranges when graph is specified"""
        args = Namespace(
            generate_all=False,
            n=None,
            rotation=None,
            graph="activation",
        )
        
        n_range, r_range, graph_types = self.command._determine_iteration_ranges(args)
        
        assert list(n_range) == list(range(2, 9))
        assert list(r_range) == list(range(6))
        assert graph_types == ["activation"]

    def test_determine_iteration_ranges_all_specified(self):
        """Test iteration ranges when all parameters are specified"""
        args = Namespace(
            generate_all=False,
            n=4,
            rotation=1,
            graph="weight",
        )
        
        n_range, r_range, graph_types = self.command._determine_iteration_ranges(args)
        
        assert list(n_range) == [4]
        assert list(r_range) == [1]
        assert graph_types == ["weight"]

    @patch('commands.reference_command.HexagonalNeuralNetwork')
    @patch('commands.reference_command.get_activation_function')
    @patch('commands.reference_command.get_loss_function')
    @patch('commands.reference_command.Path')
    def test_invoke_hex_dry_run(self, mock_path, mock_loss, mock_activation, mock_network_class):
        """Test invoke with dry-run mode for hex model"""
        # Setup mocks
        mock_net = Mock()
        mock_network_class.return_value = mock_net
        mock_activation.return_value = Mock()
        mock_loss.return_value = Mock()
        
        args = Namespace(
            model="hex",
            n=3,
            rotation=0,
            graph="activation",
            generate_all=False,
            activation="sigmoid",
            loss="mean_squared_error",
            detail="",
            dry_run=True,
            seed=42,
        )
        
        mock_path.return_value.mkdir = Mock()
        self.command.invoke(args)
        
        # In dry-run mode, graph methods should not be called
        assert not mock_net.graph_weights.called
        assert not mock_net.graph_structure.called

    @patch('commands.reference_command.HexagonalNeuralNetwork')
    @patch('commands.reference_command.get_activation_function')
    @patch('commands.reference_command.get_loss_function')
    def test_invoke_hex_generate_graph(self, mock_loss, mock_activation, mock_network_class):
        """Test invoke generates graph for hex model"""
        # Setup mocks
        mock_net = Mock()
        mock_net.n = 3
        mock_net.r = 0
        mock_net.graph_weights.return_value = ("test.png", Mock())
        mock_network_class.return_value = mock_net
        mock_activation.return_value = Mock()
        mock_loss.return_value = Mock()
        
        args = Namespace(
            model="hex",
            n=3,
            rotation=0,
            graph="activation",
            generate_all=False,
            activation="sigmoid",
            loss="mean_squared_error",
            detail="test_detail",
            dry_run=False,
            seed=42,
        )
        
        # Don't mock Path - let it use the real Path class
        # Mock mkdir on the Path object instead
        with patch('pathlib.Path.mkdir') as mock_mkdir:
            self.command.invoke(args)
        
        # Verify graph_weights was called with correct parameters
        # Use call_args to check the arguments more flexibly
        mock_net.graph_weights.assert_called_once()
        call_args = mock_net.graph_weights.call_args
        assert call_args.kwargs['activation_only'] is True
        assert call_args.kwargs['detail'] == "test_detail"
        # Check that output_dir is a Path pointing to "reference"
        output_dir = call_args.kwargs['output_dir']
        assert isinstance(output_dir, Path)
        assert str(output_dir) == "reference"

    @patch('commands.reference_command.get_available_learning_rates')
    @patch('commands.reference_command.get_learning_rate')
    @patch('commands.reference_command.FigureService')
    @patch('commands.reference_command.Path')
    def test_invoke_none_model_dry_run(self, mock_path, mock_figure_service, mock_get_lr, mock_get_available):
        """Test invoke with dry-run mode for model='none'"""
        mock_get_available.return_value = ["constant", "exponential_decay"]
        mock_lr_instance = Mock()
        mock_lr_instance.rate_at_iteration.return_value = 0.01
        mock_get_lr.return_value = mock_lr_instance
        
        args = Namespace(
            model="none",
            dry_run=True,
        )
        
        mock_path.return_value.mkdir = Mock()
        self.command.invoke(args)
        
        # In dry-run mode, figure service should not be used
        assert not mock_figure_service.return_value.init_learning_rate_ref_figure.called

    @patch('commands.reference_command.get_available_learning_rates')
    @patch('commands.reference_command.get_learning_rate')
    @patch('commands.reference_command.FigureService')
    @patch('commands.reference_command.Path')
    def test_invoke_none_model_generate(self, mock_path, mock_figure_service, mock_get_lr, mock_get_available):
        """Test invoke generates learning rate figures for model='none'"""
        mock_get_available.return_value = ["constant"]
        mock_lr_instance = Mock()
        mock_lr_instance.rate_at_iteration.return_value = 0.01
        mock_get_lr.return_value = mock_lr_instance
        
        mock_figure = Mock()
        mock_figure_service.return_value.init_learning_rate_ref_figure.return_value = mock_figure
        
        args = Namespace(
            model="none",
            dry_run=False,
        )
        
        mock_path.return_value.mkdir = Mock()
        self.command.invoke(args)
        
        # Verify figure was created and saved
        mock_figure_service.return_value.init_learning_rate_ref_figure.assert_called()
        mock_figure.update_figure.assert_called()
        mock_figure.save_figure.assert_called()

    @patch('commands.reference_command.MLPNetwork')
    @patch('commands.reference_command.get_activation_function')
    @patch('commands.reference_command.get_loss_function')
    def test_invoke_mlp_requires_graph(self, mock_loss, mock_activation, mock_mlp_class):
        """Test that MLP model requires graph to be specified"""
        args = Namespace(
            model="mlp",
            n=None,
            rotation=None,
            graph=None,
            generate_all=False,
            activation="sigmoid",
            loss="mean_squared_error",
        )
        
        with pytest.raises(ValueError, match="MLP model requires"):
            self.command.invoke(args)

    @patch('commands.reference_command.MLPNetwork')
    @patch('commands.reference_command.get_activation_function')
    @patch('commands.reference_command.get_loss_function')
    def test_invoke_mlp_generate_graph(self, mock_loss, mock_activation, mock_mlp_class):
        """Test invoke generates graph for MLP model"""
        mock_net = Mock()
        mock_net.graph_structure.return_value = "test.png"
        mock_mlp_class.return_value = mock_net
        mock_activation.return_value = Mock()
        mock_loss.return_value = Mock()
        
        args = Namespace(
            model="mlp",
            n=None,
            rotation=None,
            graph="structure_matplotlib",
            generate_all=False,
            activation="sigmoid",
            loss="mean_squared_error",
        )
        
        self.command.invoke(args)
        
        # Verify graph_structure was called
        mock_net.graph_structure.assert_called_once()

    def test_generate_graph_invalid_type(self):
        """Test that invalid graph type raises error"""
        mock_net = Mock()
        args = Namespace(
            dry_run=False,
            detail="",
            rotation=0,
        )
        figures_dir = Path("test")
        
        with pytest.raises(ValueError, match="Invalid graph type"):
            self.command._generate_graph(mock_net, "invalid_type", args, figures_dir)

    @patch('commands.reference_command.HexagonalNeuralNetwork')
    @patch('commands.reference_command.get_activation_function')
    @patch('commands.reference_command.get_loss_function')
    @patch('commands.reference_command.Path')
    def test_invoke_iterates_over_ranges(self, mock_path, mock_loss, mock_activation, mock_network_class):
        """Test that invoke iterates over all combinations when parameters are not specified"""
        mock_net = Mock()
        mock_net.n = 3
        mock_net.r = 0
        mock_net.graph_weights.return_value = ("test.png", Mock())
        mock_network_class.return_value = mock_net
        mock_activation.return_value = Mock()
        mock_loss.return_value = Mock()
        
        args = Namespace(
            model="hex",
            n=3,  # Fixed n
            rotation=None,  # Will iterate over r
            graph=None,  # Will iterate over all graph types
            generate_all=False,
            activation="sigmoid",
            loss="mean_squared_error",
            detail="",
            dry_run=False,
            seed=42,
        )
        
        mock_path.return_value.mkdir = Mock()
        self.command.invoke(args)
        
        # Should create network for each r value (0-5) and each graph type (6 types)
        # So 6 * 6 = 36 calls
        assert mock_network_class.call_count == 6  # One for each r value

    @patch('commands.reference_command.HexagonalNeuralNetwork')
    @patch('commands.reference_command.get_activation_function')
    @patch('commands.reference_command.get_loss_function')
    @patch('commands.reference_command.Path')
    def test_invoke_all_flag(self, mock_path, mock_loss, mock_activation, mock_network_class):
        """Test that --all flag generates all combinations"""
        mock_net = Mock()
        mock_net.n = 2
        mock_net.r = 0
        mock_net.graph_weights.return_value = ("test.png", Mock())
        mock_net.graph_structure.return_value = ("test.png", Mock())
        mock_net._graph_multi_activation.return_value = ("test.png", Mock())
        mock_network_class.return_value = mock_net
        mock_activation.return_value = Mock()
        mock_loss.return_value = Mock()
        
        args = Namespace(
            model="hex",
            n=None,
            rotation=None,
            graph=None,
            generate_all=True,
            activation="sigmoid",
            loss="mean_squared_error",
            detail="",
            dry_run=True,  # Use dry-run to avoid creating many files
            seed=42,
        )
        
        mock_path.return_value.mkdir = Mock()
        self.command.invoke(args)
        
        # Should iterate over n=2..8 (7 values) and r=0..5 (6 values)
        # In dry-run mode, we just verify the structure works
        assert mock_network_class.called
