from __future__ import annotations

import re
import shutil
from argparse import ArgumentParser, Namespace

from commands.command import Command
from commands.maze_game import (
    format_frame,
    in_disk,
    plot_maze_hexbin,
    step_axial,
)

_DISK_RADIUS = 2

# Substrings that indicate a GUI-capable matplotlib backend (after lowercasing).
_GUI_BACKEND_MARKERS = (
    "tkagg",
    "tkcairo",
    "qtagg",
    "qtcairo",
    "qt5agg",
    "gtk",
    "wxagg",
    "wxcairo",
    "macosx",
    "notebook",
    "nbagg",
    "webagg",
)


def _matplotlib_backend_can_show_windows() -> bool:
    import matplotlib

    name = matplotlib.get_backend().lower()
    if name.startswith("module://"):
        return False
    return any(marker in name for marker in _GUI_BACKEND_MARKERS)


def _ensure_interactive_matplotlib_backend() -> None:
    """Switch off Agg/svg/pdf backends so ``plt.pause`` / plot windows work."""
    import matplotlib.pyplot as plt

    if _matplotlib_backend_can_show_windows():
        return

    errors: list[str] = []
    for candidate in ("TkAgg", "Qt5Agg", "QtAgg", "Gtk4Agg", "Gtk3Agg", "WxAgg"):
        try:
            plt.switch_backend(candidate)
            return
        except Exception as exc:
            errors.append(f"{candidate}: {exc}")

    hint = (
        "Cannot open an interactive matplotlib window from this Python build.\n"
        "The default backend was non-interactive (often Agg after a headless import).\n"
        "Fix one of:\n"
        "  • Ubuntu/Debian: sudo apt install python3-tk   (enables TkAgg)\n"
        "  • Or install PyQt5 / PySide6 and use e.g. MPLBACKEND=Qt5Agg\n"
        "  • Or run with a desktop Python that already has Tk/Qt/GTK bindings.\n"
        "switch_backend attempts:\n  " + "\n  ".join(errors[:6])
    )
    raise RuntimeError(hint)


class MazeCommand(Command):
    show_cli_banner = False

    def name(self) -> str:
        return "maze"

    def help(self) -> str:
        return "Easter egg: tiny hex-disk walk (ant); try m0–m5 / q0–q5"

    def configure_parser(self, parser: ArgumentParser):
        parser.add_argument(
            "--window",
            default="cli",
            choices=("cli", "plot"),
            help='Display: "cli" (terminal) or "plot" (matplotlib hexbin)',
            dest="window",
        )

    def validate_args(self, args: Namespace):
        pass

    def invoke(self, args: Namespace):
        if args.window == "plot":
            try:
                self._invoke_plot(args)
            except RuntimeError as exc:
                import sys

                print(exc, file=sys.stderr)
                raise SystemExit(1) from exc
        else:
            self._invoke_cli(args)

    def _invoke_cli(self, args: Namespace):
        ant_q, ant_r = 0, 0
        moves_ok = 0
        last = "Welcome — ant at center (0,0)."
        columns = shutil.get_terminal_size((80, 24)).columns

        print(format_frame(ant_q, ant_r, _DISK_RADIUS, columns, moves_ok, last))

        move_re = re.compile(r"^m([0-5])$", re.I)
        query_re = re.compile(r"^q([0-5])$", re.I)

        while True:
            try:
                raw = input("maze> ").strip()
            except EOFError:
                print("\nBye.")
                break

            if not raw:
                last = "(empty line — type help for commands)"
                columns = shutil.get_terminal_size((80, 24)).columns
                print(format_frame(ant_q, ant_r, _DISK_RADIUS, columns, moves_ok, last))
                continue

            low = raw.lower()
            if low in ("quit", "exit"):
                print("Bye.")
                break

            if low == "help":
                last = "help: m0–m5 move toward hex_D; q0–q5 query (stub); quit or exit."
                columns = shutil.get_terminal_size((80, 24)).columns
                print(format_frame(ant_q, ant_r, _DISK_RADIUS, columns, moves_ok, last))
                continue

            if m := move_re.match(low):
                d = int(m.group(1))
                nq, nr = step_axial(ant_q, ant_r, d)
                if not in_disk(nq, nr, _DISK_RADIUS):
                    last = f"m{d} blocked — ({nq},{nr}) is outside the disk."
                else:
                    ant_q, ant_r = nq, nr
                    moves_ok += 1
                    last = f"m{d} → ant now at ({ant_q},{ant_r})."
                columns = shutil.get_terminal_size((80, 24)).columns
                print(format_frame(ant_q, ant_r, _DISK_RADIUS, columns, moves_ok, last))
                continue

            if m := query_re.match(low):
                d = int(m.group(1))
                last = f"Query hex_{d}: (not implemented yet)."
                columns = shutil.get_terminal_size((80, 24)).columns
                print(format_frame(ant_q, ant_r, _DISK_RADIUS, columns, moves_ok, last))
                continue

            last = f"Unknown command {raw!r} — try help."
            columns = shutil.get_terminal_size((80, 24)).columns
            print(format_frame(ant_q, ant_r, _DISK_RADIUS, columns, moves_ok, last))

    def _invoke_plot(self, args: Namespace):
        import matplotlib.pyplot as plt

        _ensure_interactive_matplotlib_backend()

        ant_q, ant_r = 0, 0
        moves_ok = 0
        last = "Welcome — ant at center (0,0)."
        plt.ion()
        fig, ax = plt.subplots(figsize=(7, 6))
        fig.subplots_adjust(top=0.88)
        try:
            fig.canvas.manager.set_window_title("hexnet maze")
        except AttributeError:
            pass

        _plot_shown = {"done": False}

        def refresh_ui():
            plot_maze_hexbin(ax, ant_q, ant_r, _DISK_RADIUS)
            fig.suptitle(
                f"Hex maze · r={_DISK_RADIUS} · ant ({ant_q},{ant_r}) · moves {moves_ok} · {last}",
                fontsize=11,
            )
            fig.canvas.draw_idle()
            if not _plot_shown["done"]:
                plt.show(block=False)
                _plot_shown["done"] = True
            fig.canvas.flush_events()
            plt.pause(0.05)

        refresh_ui()

        move_re = re.compile(r"^m([0-5])$", re.I)
        query_re = re.compile(r"^q([0-5])$", re.I)

        try:
            while True:
                try:
                    raw = input("maze> ").strip()
                except EOFError:
                    print("\nBye.")
                    break

                if not raw:
                    last = "(empty line — type help for commands)"
                    refresh_ui()
                    continue

                low = raw.lower()
                if low in ("quit", "exit"):
                    print("Bye.")
                    break

                if low == "help":
                    last = "help: m0–m5 move toward hex_D; q0–q5 query (stub); quit or exit."
                    refresh_ui()
                    continue

                if m := move_re.match(low):
                    d = int(m.group(1))
                    nq, nr = step_axial(ant_q, ant_r, d)
                    if not in_disk(nq, nr, _DISK_RADIUS):
                        last = f"m{d} blocked — ({nq},{nr}) is outside the disk."
                    else:
                        ant_q, ant_r = nq, nr
                        moves_ok += 1
                        last = f"m{d} → ant now at ({ant_q},{ant_r})."
                    refresh_ui()
                    continue

                if m := query_re.match(low):
                    d = int(m.group(1))
                    last = f"Query hex_{d}: (not implemented yet)."
                    refresh_ui()
                    continue

                last = f"Unknown command {raw!r} — try help."
                refresh_ui()
        finally:
            plt.close(fig)
            plt.ioff()
