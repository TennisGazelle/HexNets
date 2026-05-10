"""Maze easter egg geometry/render tests (live under tests/ root to avoid `commands` package shadowing)."""

import argparse

from commands.maze_command import MazeCommand
from commands.maze_game import (
    AXIAL_STEP,
    axial_distance,
    axial_to_plot_xy,
    format_frame,
    in_disk,
    iter_disk_cells,
    oddq_col,
    plot_maze_hexbin,
    render_narrow,
    render_wide,
    step_axial,
)


class TestMazeGeometry:
    def test_axial_distance_origin(self):
        assert axial_distance(0, 0) == 0
        assert axial_distance(2, 0) == 2

    def test_disk_radius_2_count(self):
        cells = iter_disk_cells(2)
        assert len(cells) == 19

    def test_step_roundtrip_variants(self):
        q, r = 0, 0
        for d in range(6):
            nq, nr = step_axial(q, r, d)
            dq, dr = AXIAL_STEP[d]
            assert (nq, nr) == (dq, dr)

    def test_hex_zero_is_up_decreasing_r(self):
        assert step_axial(0, 0, 0) == (0, -1)

    def test_axial_to_plot_xy_terminal_up_matches_smaller_r_higher_y(self):
        _, y0 = axial_to_plot_xy(0, 0)
        _, y1 = axial_to_plot_xy(*step_axial(0, 0, 0))
        assert y1 > y0

    def test_plot_maze_hexbin_smoke(self):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        try:
            plot_maze_hexbin(ax, 0, 0, 2)
        finally:
            plt.close(fig)

    def test_blocked_outside_disk(self):
        assert in_disk(0, 0, 2)
        assert not in_disk(3, 0, 2)

    def test_oddq_col_injective_on_disk(self):
        cells = iter_disk_cells(2)
        seen = set()
        for q, r in cells:
            key = (oddq_col(q, r), r)
            assert key not in seen
            seen.add(key)


class TestMazeRender:
    def test_format_frame_wide_contains_ant(self):
        text = format_frame(0, 0, 2, 120, 0, "ok")
        assert "@" in text
        assert "ok" in text

    def test_format_frame_narrow_skips_tiling(self):
        text = format_frame(1, -1, 2, 79, 3, "moved")
        assert "@" not in text
        assert "moved" in text
        assert "hex_0" in text

    def test_render_narrow_lists_hex_dirs(self):
        text = render_narrow(0, 0, 2)
        assert "m0" in text and "q0" in text

    def test_render_wide_large_radius_one_row_per_axial_r(self):
        text = render_wide(0, 0, 5)
        dot_rows = [ln for ln in text.splitlines() if ln.strip()]
        assert len(dot_rows) == 11
        assert "@" in text


class TestMazeCommandParser:
    def test_configure_parser_smoke(self):
        cmd = MazeCommand()
        parser = argparse.ArgumentParser()
        cmd.configure_parser(parser)
        args = parser.parse_args([])
        cmd.validate_args(args)
        assert args.window == "cli"

    def test_window_plot_choice(self):
        cmd = MazeCommand()
        parser = argparse.ArgumentParser()
        cmd.configure_parser(parser)
        args = parser.parse_args(["--window", "plot"])
        assert args.window == "plot"
