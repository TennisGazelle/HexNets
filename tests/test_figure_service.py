"""Unit tests for figure_service.py"""

import pytest
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
)


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

    @patch('services.figure_service.FigureService.LearningRateRefFigure')
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

    @patch('services.figure_service.FigureService.TrainingFigure')
    def test_init_training_figure(self, mock_figure_class):
        """Test initializing training figure"""
        mock_figure = Mock()
        mock_figure_class.return_value = mock_figure
        
        filename = "test_training.png"
        title = "Test Training"
        loss_detail = "MSE"
        accuracy_detail = "RMSE"
        r2_detail = "R^2"
        
        result = self.service.init_training_figure(
            filename, title, loss_detail, accuracy_detail, r2_detail
        )
        
        assert result == mock_figure
        mock_figure_class.assert_called_once_with(
            title,
            self.service.figures_path / filename,
            loss_detail,
            accuracy_detail,
            r2_detail
        )
        assert self.service.figures[title] == mock_figure

    @patch('services.figure_service.FigureService.RefFigure')
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
            mock_ax_acc = Mock()
            mock_ax_r2 = Mock()
            mock_ax_adj_r2 = Mock()
            mock_subplots.return_value = (mock_fig, (mock_ax_loss, mock_ax_acc, mock_ax_r2, mock_ax_adj_r2))
            
            # Mock plot lines
            mock_line = Mock()
            mock_ax_loss.plot.return_value = (mock_line,)
            mock_ax_acc.plot.return_value = (mock_line,)
            mock_ax_r2.plot.return_value = (mock_line,)
            mock_ax_adj_r2.plot.return_value = (mock_line,)
            
            self.filename = "test_training.png"
            self.title = "Test Training"
            self.loss_detail = "MSE"
            self.accuracy_detail = "RMSE"
            self.r2_detail = "R^2"
            
            self.figure = TrainingFigure(
                self.title, self.filename, self.loss_detail,
                self.accuracy_detail, self.r2_detail
            )
            self.figure.fig = mock_fig
            self.figure.ax_loss = mock_ax_loss
            self.figure.ax_acc = mock_ax_acc
            self.figure.ax_r2 = mock_ax_r2
            self.figure.ax_adj_r2 = mock_ax_adj_r2

    def test_init(self):
        """Test that TrainingFigure initializes correctly"""
        assert self.figure.filename == self.filename
        assert self.figure.title == self.title
        assert self.figure.loss_detail == self.loss_detail
        assert self.figure.accuracy_detail == self.accuracy_detail
        assert self.figure.r2_detail == self.r2_detail
        assert len(self.figure.training_metrics) == 6  # 6 channels

    def test_init_channels(self):
        """Test that all channels are initialized"""
        for channel in range(6):
            assert channel in self.figure.training_metrics
            assert "loss" in self.figure.training_metrics[channel]
            assert "accuracy" in self.figure.training_metrics[channel]
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
            "accuracy": 0.8,
            "r_squared": 0.9,
            "adjusted_r_squared": 0.85
        }
        channel = 0
        
        self.figure.update_figure(training_metrics, channel)
        
        assert len(self.figure.training_metrics[channel]["loss"]) == 1
        assert len(self.figure.training_metrics[channel]["accuracy"]) == 1
        assert len(self.figure.training_metrics[channel]["r_squared"]) == 1
        assert len(self.figure.training_metrics[channel]["adjusted_r_squared"]) == 1
        assert self.figure.training_metrics[channel]["loss"][0] == 0.5
        assert self.figure.training_metrics[channel]["accuracy"][0] == 0.8
        assert self.figure.training_metrics[channel]["r_squared"][0] == 0.9
        assert self.figure.training_metrics[channel]["adjusted_r_squared"][0] == 0.85

    def test_update_figure_incomplete_metrics(self):
        """Test updating figure with incomplete metrics (should not update)"""
        training_metrics = {
            "loss": 0.5,
            # Missing accuracy and r_squared - implementation returns early without updating
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
            "accuracy": [],
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
            "accuracy": 0.8,
            "r_squared": 0.9,
            "adjusted_r_squared": 0.85
        }
        training_metrics_1 = {
            "loss": 0.3,
            "accuracy": 0.9,
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
            "accuracy": 0.8,
            "r_squared": 0.9,
            "adjusted_r_squared": 0.85
        }
        training_metrics_2 = {
            "loss": 0.4,
            "accuracy": 0.85,
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
        
        with patch('services.figure_service.FigureService.LearningRateRefFigure') as mock_figure_class:
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
