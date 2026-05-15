"""Unit tests for figure_service.py"""

import matplotlib
matplotlib.use("Agg")

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import matplotlib.pyplot as plt
import tempfile

from services.figure_service import (
    FigureService,
    LearningRateRefFigure,
    TrainingFigure,
    RefFigure,
    DynamicWeightsFigure,
)
from services.figure_service.FigureService import REGRESSION_SCORE_DETAIL, R2_DETAIL

# Patch the module where FigureService is defined (FigureService.py). The package
# re-exports the class as FigureService, so 'services.figure_service.FigureService'
# resolves to the class on GHA (Python 3.10), not the module — use the module explicitly.
FIGURE_SERVICE_MODULE = sys.modules["services.figure_service.FigureService"]


class TestFigureService:
    """Test cases for FigureService"""

    def setup_method(self):
        """Set up test fixtures"""
        self.service = FigureService()

    def test_init(self):
        """Test that FigureService initializes correctly"""
        assert self.service.figures_path == Path("figures")
        assert isinstance(self.service.figures, dict)
        assert len(self.service.figures) == 0

    def test_set_figures_path(self):
        """Test setting figures path"""
        new_path = Path("test_figures")
        self.service.set_figures_path(new_path)
        assert self.service.figures_path == new_path

    def test_set_figures_path_none(self):
        """Test setting figures path to None uses default"""
        self.service.set_figures_path(None)
        assert self.service.figures_path == Path("figures")

    @patch.object(FIGURE_SERVICE_MODULE, "LearningRateRefFigure")
    def test_init_learning_rate_ref_figure(self, mock_figure_class):
        """Test initializing learning rate reference figure"""
        mock_figure = Mock()
        mock_figure_class.return_value = mock_figure
        
        filename = "test_lr.png"
        title = "Test LR"
        lr_name = "constant"
        max_iterations = 500
        
        result = self.service.init_learning_rate_ref_figure(
            filename, title, lr_name, max_iterations
        )
        
        assert result == mock_figure
        mock_figure_class.assert_called_once_with(
            title,
            self.service.figures_path / filename,
            lr_name,
            max_iterations
        )
        assert self.service.figures[title] == mock_figure

    @patch.object(FIGURE_SERVICE_MODULE, "TrainingFigure")
    def test_init_training_figure(self, mock_figure_class):
        """Test initializing training figure"""
        mock_figure = Mock()
        mock_figure_class.return_value = mock_figure
        
        filename = "test_training.png"
        title = "Test Training"
        loss_detail = "MSE"
        regression_score_detail = "RMSE"
        r2_detail = "R^2"
        
        result = self.service.init_training_figure(
            filename, title, loss_detail, regression_score_detail, r2_detail
        )
        
        assert result == mock_figure
        mock_figure_class.assert_called_once_with(
            title,
            self.service.figures_path / filename,
            loss_detail,
            regression_score_detail,
            r2_detail
        )
        assert self.service.figures[title] == mock_figure

    @patch.object(FIGURE_SERVICE_MODULE, "RefFigure")
    def test_init_ref_figure(self, mock_figure_class):
        """Test initializing reference figure"""
        mock_figure = Mock()
        mock_figure_class.return_value = mock_figure
        
        filename = "test_ref.png"
        title = "Test Ref"
        detail = "test detail"
        
        result = self.service.init_ref_figure(filename, title, detail)
        
        assert result == mock_figure
        mock_figure_class.assert_called_once_with(title, filename, detail)
        assert self.service.figures[title] == mock_figure


class TestLearningRateRefFigure:
    """Test cases for LearningRateRefFigure"""

    def setup_method(self):
        """Set up test fixtures"""
        with patch('matplotlib.pyplot.subplots') as mock_subplots:
            mock_fig = Mock()
            mock_ax = Mock()
            mock_line = Mock()
            mock_ax.plot.return_value = (mock_line,)
            mock_subplots.return_value = (mock_fig, mock_ax)
            
            self.filename = "test_lr.png"
            self.title = "Test Learning Rate"
            self.lr_name = "constant"
            self.max_iterations = 500
            
            self.figure = LearningRateRefFigure(
                self.title, self.filename, self.lr_name, self.max_iterations
            )
            self.figure.fig = mock_fig
            self.figure.ax = mock_ax
            self.figure.line = mock_line

    def test_init(self):
        """Test that LearningRateRefFigure initializes correctly"""
        assert self.figure.filename == self.filename
        assert self.figure.title == self.title
        assert self.figure.learning_rate_name == self.lr_name
        assert self.figure.max_iterations == self.max_iterations

    def test_save_figure(self):
        """Test saving figure"""
        # Use a temporary directory for testing
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "test.png"
            self.figure.filename = str(test_file)
            # Mock the figure's savefig to avoid actually creating files
            with patch.object(self.figure.fig, 'savefig') as mock_savefig:
                self.figure.save_figure()
                mock_savefig.assert_called_once()
                # Verify the directory was created
                assert test_file.parent.exists()

    def test_save_figure_path_object(self):
        """Test saving figure when filename is already a Path object"""
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            path_obj = Path(tmpdir) / "test.png"
            self.figure.filename = path_obj
            
            with patch.object(self.figure.fig, 'savefig') as mock_savefig:
                self.figure.save_figure()
                mock_savefig.assert_called_once()
                # Verify the directory was created
                assert path_obj.parent.exists()

    def test_update_figure(self):
        """Test updating figure with data"""
        iterations = np.array([1, 2, 3, 4, 5])
        lr_values = np.array([0.01, 0.01, 0.01, 0.01, 0.01])
        
        self.figure.update_figure(iterations, lr_values)
        
        self.figure.line.set_data.assert_called_once_with(iterations, lr_values)
        self.figure.ax.relim.assert_called_once()
        self.figure.ax.autoscale_view.assert_called_once()
        self.figure.fig.canvas.draw.assert_called_once()

    def test_update_figure_empty_arrays(self):
        """Test updating figure with empty arrays"""
        iterations = np.array([])
        lr_values = np.array([])
        
        self.figure.update_figure(iterations, lr_values)
        
        self.figure.line.set_data.assert_called_once_with(iterations, lr_values)

    def test_show_figure(self):
        """Test showing figure"""
        self.figure.show_figure()
        self.figure.fig.show.assert_called_once()


class TestTrainingFigure:
    """Test cases for TrainingFigure"""

    def setup_method(self):
        """Set up test fixtures"""
        with patch('matplotlib.pyplot.subplots') as mock_subplots:
            mock_fig = Mock()
            mock_ax_loss = Mock()
            mock_ax_reg_score = Mock()
            mock_ax_r2 = Mock()
            mock_ax_adj_r2 = Mock()
            mock_subplots.return_value = (mock_fig, (mock_ax_loss, mock_ax_reg_score, mock_ax_r2, mock_ax_adj_r2))
            
            # Mock plot lines
            mock_line = Mock()
            mock_ax_loss.plot.return_value = (mock_line,)
            mock_ax_reg_score.plot.return_value = (mock_line,)
            mock_ax_r2.plot.return_value = (mock_line,)
            mock_ax_adj_r2.plot.return_value = (mock_line,)
            
            self.filename = "test_training.png"
            self.title = "Test Training"
            self.loss_detail = "MSE"
            self.regression_score_detail = "RMSE"
            self.r2_detail = "R^2"
            
            self.figure = TrainingFigure(
                self.title, self.filename, self.loss_detail,
                self.regression_score_detail, self.r2_detail
            )
            self.figure.fig = mock_fig
            self.figure.ax_loss = mock_ax_loss
            self.figure.ax_reg_score = mock_ax_reg_score
            self.figure.ax_r2 = mock_ax_r2
            self.figure.ax_adj_r2 = mock_ax_adj_r2

    def test_init(self):
        """Test that TrainingFigure initializes correctly"""
        assert self.figure.filename == self.filename
        assert self.figure.title == self.title
        assert self.figure.loss_detail == self.loss_detail
        assert self.figure.regression_score_detail == self.regression_score_detail
        assert self.figure.r2_detail == self.r2_detail
        assert len(self.figure.training_metrics) == 6  # 6 channels

    def test_init_channels(self):
        """Test that all channels are initialized"""
        for channel in range(6):
            assert channel in self.figure.training_metrics
            assert "loss" in self.figure.training_metrics[channel]
            assert "regression_score" in self.figure.training_metrics[channel]
            assert "r_squared" in self.figure.training_metrics[channel]
            assert "adjusted_r_squared" in self.figure.training_metrics[channel]

    def test_save_figure(self):
        """Test saving training figure"""
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "test.png"
            self.figure.filename = str(test_file)
            with patch.object(self.figure.fig, 'savefig') as mock_savefig:
                self.figure.save_figure()
                mock_savefig.assert_called_once()
                # Verify the directory was created
                assert test_file.parent.exists()

    def test_update_figure_complete_metrics(self):
        """Test updating figure with complete metrics"""
        training_metrics = {
            "loss": 0.5,
            "regression_score": 0.8,
            "r_squared": 0.9,
            "adjusted_r_squared": 0.85
        }
        channel = 0
        
        self.figure.update_figure(training_metrics, channel)
        
        assert len(self.figure.training_metrics[channel]["loss"]) == 1
        assert len(self.figure.training_metrics[channel]["regression_score"]) == 1
        assert len(self.figure.training_metrics[channel]["r_squared"]) == 1
        assert len(self.figure.training_metrics[channel]["adjusted_r_squared"]) == 1
        assert self.figure.training_metrics[channel]["loss"][0] == 0.5
        assert self.figure.training_metrics[channel]["regression_score"][0] == 0.8
        assert self.figure.training_metrics[channel]["r_squared"][0] == 0.9
        assert self.figure.training_metrics[channel]["adjusted_r_squared"][0] == 0.85

    def test_update_figure_incomplete_metrics(self):
        """Test updating figure with incomplete metrics (should not update)"""
        training_metrics = {
            "loss": 0.5,
            # Missing regression_score and r_squared - implementation returns early without updating
        }
        channel = 0

        initial_length = len(self.figure.training_metrics[channel]["loss"])
        self.figure.update_figure(training_metrics, channel)

        # Should not have updated (implementation returns early when required keys are missing)
        assert len(self.figure.training_metrics[channel]["loss"]) == initial_length

    def test_update_figure_empty_metrics(self):
        """Test updating figure with empty metrics"""
        training_metrics = {
            "loss": [],
            "regression_score": [],
            "r_squared": []
        }
        channel = 0
        
        initial_length = len(self.figure.training_metrics[channel]["loss"])
        self.figure.update_figure(training_metrics, channel)
        
        # Should not have updated
        assert len(self.figure.training_metrics[channel]["loss"]) == initial_length

    def test_update_figure_multiple_channels(self):
        """Test updating figure for different channels"""
        training_metrics_0 = {
            "loss": 0.5,
            "regression_score": 0.8,
            "r_squared": 0.9,
            "adjusted_r_squared": 0.85
        }
        training_metrics_1 = {
            "loss": 0.3,
            "regression_score": 0.9,
            "r_squared": 0.95,
            "adjusted_r_squared": 0.92
        }
        
        self.figure.update_figure(training_metrics_0, channel=0)
        self.figure.update_figure(training_metrics_1, channel=1)
        
        assert self.figure.training_metrics[0]["loss"][0] == 0.5
        assert self.figure.training_metrics[1]["loss"][0] == 0.3
        assert len(self.figure.training_metrics[0]["loss"]) == 1
        assert len(self.figure.training_metrics[1]["loss"]) == 1

    def test_update_figure_multiple_updates(self):
        """Test updating figure multiple times for same channel"""
        training_metrics_1 = {
            "loss": 0.5,
            "regression_score": 0.8,
            "r_squared": 0.9,
            "adjusted_r_squared": 0.85
        }
        training_metrics_2 = {
            "loss": 0.4,
            "regression_score": 0.85,
            "r_squared": 0.92,
            "adjusted_r_squared": 0.88
        }
        
        self.figure.update_figure(training_metrics_1, channel=0)
        self.figure.update_figure(training_metrics_2, channel=0)
        
        assert len(self.figure.training_metrics[0]["loss"]) == 2
        assert self.figure.training_metrics[0]["loss"][0] == 0.5
        assert self.figure.training_metrics[0]["loss"][1] == 0.4

    def test_show_figure(self):
        """Test showing training figure"""
        self.figure.show_figure()
        self.figure.fig.show.assert_called_once()


class TestRefFigure:
    """Test cases for RefFigure"""

    def setup_method(self):
        """Set up test fixtures"""
        with patch('matplotlib.pyplot.figure') as mock_figure:
            mock_fig = Mock()
            mock_figure.return_value = mock_fig
            
            self.filename = "test_ref.png"
            self.title = "Test Ref"
            self.detail = "test detail"
            
            self.figure = RefFigure(self.title, self.filename, self.detail)
            self.figure.fig = mock_fig

    def test_init(self):
        """Test that RefFigure initializes correctly"""
        assert self.figure.filename == self.filename
        assert self.figure.title == self.title

    def test_save_figure(self):
        """Test saving reference figure"""
        # RefFigure.save_figure() calls self.fig.savefig() without arguments
        # This is a simple implementation
        self.figure.save_figure()
        self.figure.fig.savefig.assert_called_once()

    def test_show_figure(self):
        """Test showing reference figure"""
        self.figure.show_figure()
        self.figure.fig.show.assert_called_once()


class TestFigureServiceEdgeCases:
    """Test edge cases for FigureService"""

    def test_multiple_figures_same_title(self):
        """Test that creating multiple figures with same title overwrites"""
        service = FigureService()
        
        with patch.object(FIGURE_SERVICE_MODULE, "LearningRateRefFigure") as mock_figure_class:
            mock_figure1 = Mock()
            mock_figure2 = Mock()
            mock_figure_class.side_effect = [mock_figure1, mock_figure2]
            
            service.init_learning_rate_ref_figure("test.png", "Same Title", "constant", 500)
            assert service.figures["Same Title"] == mock_figure1
            
            service.init_learning_rate_ref_figure("test2.png", "Same Title", "exponential", 500)
            assert service.figures["Same Title"] == mock_figure2
            assert len(service.figures) == 1  # Only one entry

    def test_figures_path_with_relative_path(self):
        """Test setting figures path with relative path string"""
        service = FigureService()
        service.set_figures_path("relative/path")
        # set_figures_path converts strings to Path objects
        assert isinstance(service.figures_path, Path)
        print(service.figures_path, service.figures_path.name)
        assert str(service.figures_path) == "relative/path"

    def test_figures_path_with_absolute_path(self):
        """Test setting figures path with absolute path"""
        service = FigureService()
        abs_path = Path("/absolute/path/to/figures")
        service.set_figures_path(abs_path)
        assert service.figures_path == abs_path


class TestPrepareTrainingAnimation:
    """Tests for FigureService.prepare_training_animation."""

    def _make_loss(self, name="mean_squared_error"):
        loss = Mock()
        loss.display_name = name
        loss.__str__ = lambda self: name
        return loss

    def _make_activation(self, name="sigmoid"):
        act = Mock()
        act.display_name = name
        return act

    def _make_training_figure(self, tmp_path):
        service = FigureService()
        service.set_figures_path(tmp_path)
        return service.init_training_figure(
            "mlpnet_training_mse_sigmoid.png",
            "Initial title",
            "mse",
            REGRESSION_SCORE_DETAIL,
            R2_DETAIL,
        )

    def test_simple_names_uses_stable_basename(self, tmp_path):
        service = FigureService()
        tf = self._make_training_figure(tmp_path)
        loss = self._make_loss()
        act = self._make_activation()

        service.prepare_training_animation(
            tf,
            output_dir=tmp_path / "plots",
            simple_names=True,
            network_kind="MLP",
            display_name="mlp",
            loss=loss,
            activation=act,
        )

        assert tf.filename.name == "training_metrics.png"
        assert tf.filename.parent == tmp_path / "plots"

    def test_descriptive_names_preserves_original_basename(self, tmp_path):
        service = FigureService()
        tf = self._make_training_figure(tmp_path)
        loss = self._make_loss()
        act = self._make_activation()

        service.prepare_training_animation(
            tf,
            output_dir=tmp_path / "plots",
            simple_names=False,
            network_kind="MLP",
            display_name="mlp",
            loss=loss,
            activation=act,
        )

        assert tf.filename.name == "mlpnet_training_mse_sigmoid.png"

    def test_title_contains_network_kind_and_display_name(self, tmp_path):
        service = FigureService()
        tf = self._make_training_figure(tmp_path)
        loss = self._make_loss("huber")
        act = self._make_activation("relu")

        service.prepare_training_animation(
            tf,
            output_dir=tmp_path / "plots",
            simple_names=True,
            network_kind="Hexagonal",
            display_name="hex",
            loss=loss,
            activation=act,
        )

        assert "Hexagonal" in tf.title
        assert "hex" in tf.title
        assert "huber" in tf.title
        assert "relu" in tf.title

    def test_detail_strings_use_shared_constants(self, tmp_path):
        service = FigureService()
        tf = self._make_training_figure(tmp_path)
        loss = self._make_loss()
        act = self._make_activation()

        service.prepare_training_animation(
            tf,
            output_dir=tmp_path / "plots",
            simple_names=True,
            network_kind="MLP",
            display_name="mlp",
            loss=loss,
            activation=act,
        )

        assert tf.regression_score_detail == REGRESSION_SCORE_DETAIL
        assert tf.r2_detail == R2_DETAIL

    def test_init_weights_live_figure_creates_figure(self, tmp_path):
        service = FigureService()
        service.set_figures_path(tmp_path)
        layer_shapes = [(3, 4), (4, 2)]
        wf = service.init_weights_live_figure("weights_live.png", "Live Weights", layer_shapes)
        key = FigureService.weights_live_figure_key(layer_shapes)

        assert wf is service.figures[key]
        assert isinstance(wf, DynamicWeightsFigure)
        assert wf.filename == tmp_path / "weights_live.png"

    def test_init_weights_live_figure_reuses_when_layer_shapes_match(self, tmp_path):
        service = FigureService()
        service.set_figures_path(tmp_path)
        layer_shapes = [(3, 4)]
        key = FigureService.weights_live_figure_key(layer_shapes)
        wf1 = service.init_weights_live_figure("first.png", "First title", layer_shapes)
        n_figs = len(plt.get_fignums())
        wf2 = service.init_weights_live_figure("second.png", "Second title", layer_shapes)
        assert wf1 is wf2
        assert wf2 is service.figures[key]
        assert wf2.title == "Second title"
        assert wf2.filename == tmp_path / "second.png"
        assert len(plt.get_fignums()) == n_figs
        plt.close(wf2.fig)

    def test_init_weights_live_figure_replaces_when_layer_shapes_differ(self, tmp_path):
        service = FigureService()
        service.set_figures_path(tmp_path)
        k_small = FigureService.weights_live_figure_key([(2, 2)])
        k_large = FigureService.weights_live_figure_key([(5, 5)])
        wf_small = service.init_weights_live_figure("w.png", "A", [(2, 2)])
        old_fig = wf_small.fig
        wf_large = service.init_weights_live_figure("w.png", "B", [(5, 5)])
        assert wf_small is not wf_large
        assert k_small not in service.figures
        assert service.figures[k_large] is wf_large
        assert wf_large.layer_shapes == ((5, 5),)
        assert wf_large.fig is not old_fig
        plt.close(wf_large.fig)


class TestDynamicWeightsFigure:
    """Tests for DynamicWeightsFigure."""

    def test_init_single_layer(self, tmp_path):
        shapes = [(5, 5)]
        fig = DynamicWeightsFigure("title", tmp_path / "w.png", shapes)
        assert len(fig.images) == 1
        assert fig.filename == tmp_path / "w.png"
        plt.close("all")

    def test_init_multi_layer(self, tmp_path):
        shapes = [(3, 4), (4, 2), (2, 1)]
        fig = DynamicWeightsFigure("title", tmp_path / "w.png", shapes)
        assert len(fig.images) == 3
        plt.close("all")

    def test_update_figure_raw_weights(self, tmp_path):
        shapes = [(3, 4)]
        fig = DynamicWeightsFigure("title", tmp_path / "w.png", shapes)
        W = np.random.randn(3, 4)
        fig.update_figure([W])
        data = fig.images[0].get_array()
        assert data.shape == (3, 4)
        np.testing.assert_array_equal(data, W)
        plt.close("all")

    def test_update_figure_activation_only(self, tmp_path):
        shapes = [(4, 4)]
        fig = DynamicWeightsFigure("title", tmp_path / "w.png", shapes)
        W = np.array([[1.0, 0.0, -0.5, 0.0], [0.0, 2.0, 0.0, 0.0],
                      [0.0, 0.0, 0.3, 0.0], [0.0, 0.0, 0.0, 0.0]])
        fig.update_figure([W], activation_only=True)
        data = fig.images[0].get_array()
        expected = (W != 0).astype(float)
        np.testing.assert_array_equal(data, expected)
        plt.close("all")

    def test_save_figure_creates_file(self, tmp_path):
        shapes = [(3, 3)]
        fig = DynamicWeightsFigure("title", tmp_path / "w.png", shapes)
        W = np.eye(3)
        fig.update_figure([W])
        fig.save_figure()
        assert (tmp_path / "w.png").is_file()
        assert (tmp_path / "w.png").stat().st_size > 0
        plt.close("all")

    def test_layer_shapes_from_matrices(self):
        matrices = [np.zeros((3, 4)), np.zeros((4, 2))]
        shapes = DynamicWeightsFigure.layer_shapes_from_matrices(matrices)
        assert shapes == [(3, 4), (4, 2)]

    def test_update_figure_highlight_masks_smoke(self, tmp_path):
        shapes = [(4, 4)]
        fig = DynamicWeightsFigure("title", tmp_path / "w.png", shapes)
        W = np.random.randn(4, 4)
        mask = np.zeros((4, 4), dtype=bool)
        mask[1, 2] = True
        mask[3, 0] = True
        fig.update_figure([W], highlight_masks=[mask])
        assert len(fig._highlight_patches[0]) == 2
        plt.close("all")

    def test_highlight_edgecolor_matches_training_channel_colors(self, tmp_path):
        """Same tab10 edge color for all cells; changes with highlight_channel like TrainingFigure."""
        shapes = [(2, 2)]
        fig = DynamicWeightsFigure("title", tmp_path / "w.png", shapes)
        W = np.ones((2, 2), dtype=float)
        mask = np.ones((2, 2), dtype=bool)
        fig.update_figure([W], highlight_masks=[mask], highlight_channel=0)
        patches = fig._highlight_patches[0]
        c0 = np.asarray(patches[0].get_edgecolor()).ravel()
        for p in patches[1:]:
            np.testing.assert_allclose(np.asarray(p.get_edgecolor()).ravel(), c0)
        fig.update_figure([W], highlight_masks=[mask], highlight_channel=3)
        c3 = np.asarray(fig._highlight_patches[0][0].get_edgecolor()).ravel()
        assert not np.allclose(c0[:3], c3[:3], atol=0.05)
        plt.close("all")

    def test_update_figure_highlight_masks_cleared_on_redraw(self, tmp_path):
        shapes = [(3, 3)]
        fig = DynamicWeightsFigure("title", tmp_path / "w.png", shapes)
        W = np.eye(3)
        m1 = np.zeros((3, 3), dtype=bool)
        m1[0, 1] = True
        fig.update_figure([W], highlight_masks=[m1])
        assert len(fig._highlight_patches[0]) == 1
        m2 = np.zeros((3, 3), dtype=bool)
        m2[2, 2] = True
        fig.update_figure([W], highlight_masks=[m2])
        assert len(fig._highlight_patches[0]) == 1
        plt.close("all")
