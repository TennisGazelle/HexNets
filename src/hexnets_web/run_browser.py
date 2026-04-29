import json
import pathlib

import streamlit as st

# Align with RunService.runs_dir (process cwd, typically repo root).
RUNS_DIR = pathlib.Path("runs/").resolve()


def _list_run_directories() -> list[pathlib.Path]:
    if not RUNS_DIR.is_dir():
        return []
    dirs = [p for p in RUNS_DIR.iterdir() if p.is_dir()]
    dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return dirs


def _json_artifacts(run_path: pathlib.Path) -> list[str]:
    if not run_path.is_dir():
        return []
    preferred = ("config.json", "manifest.json", "training_metrics.json")
    names = {p.name for p in run_path.glob("*.json") if p.is_file()}
    ordered = [n for n in preferred if n in names]
    ordered.extend(sorted(names - set(ordered)))
    return ordered


def _training_plot_paths(run_path: pathlib.Path) -> list[pathlib.Path]:
    plots_dir = run_path / "plots"
    if not plots_dir.is_dir():
        return []
    # Hex: hexnet_training_*.png — MLP: mlpnet_training_*.png (same plots/ convention as RunService).
    return sorted(plots_dir.glob("*net_training_*.png"))


def _render_json_viewer_section(run_path: pathlib.Path, json_files: list[str]) -> None:
    st.subheader("JSON viewer")
    if not json_files:
        st.info("No `.json` files in this run folder root.")
        return
    if st.session_state.get("run_browser_selected_json") not in json_files:
        st.session_state.run_browser_selected_json = json_files[0]
    st.selectbox("JSON file", options=json_files, key="run_browser_selected_json")
    _render_json_viewer(run_path, st.session_state.run_browser_selected_json)


def _render_json_viewer(run_path: pathlib.Path, relative_name: str) -> None:
    path = run_path / relative_name
    if not path.is_file():
        st.warning(f"Missing file: {relative_name}")
        return
    raw = path.read_text(encoding="utf-8", errors="replace")
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        st.caption("Could not parse as JSON — showing raw text.")
        st.code(raw, language="json")
        return
    st.json(parsed)


def _set_run_browser_selected_run(run_name: str) -> None:
    """Session-state only; used as `st.button(..., on_click=...)` so selection updates before the tree re-renders."""
    st.session_state.run_browser_selected_run = run_name


def _render_run_tree(run_dirs: list[pathlib.Path]) -> None:
    st.subheader("Runs tree (read-only)")
    for d in run_dirs:
        is_selected = d.name == st.session_state.run_browser_selected_run
        label = f"{d.name} · selected" if is_selected else d.name
        with st.expander(label, expanded=is_selected):
            if is_selected:
                st.success("This run is selected.")
            entries = sorted(d.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
            if not entries:
                st.caption("(empty)")
            else:
                for child in entries:
                    if child.is_dir():
                        st.markdown(f"- **{child.name}/**")
                        if child.name == "plots":
                            pngs = sorted(child.glob("*.png"))
                            for png in pngs[:40]:
                                st.caption(f"  - {png.name}")
                            if len(pngs) > 40:
                                st.caption(f"  - … and {len(pngs) - 40} more PNGs")
                    else:
                        st.markdown(f"- `{child.name}`")

            st.button(
                "Use this directory",
                key=f"run_browser_select_{d.name}",
                disabled=is_selected,
                on_click=_set_run_browser_selected_run,
                kwargs={"run_name": d.name},
            )


def _render_training_plots(paths: list[pathlib.Path]) -> None:
    """Plot column only; caller checks paths non-empty."""
    st.subheader("Training plot")
    for p in paths:
        st.caption(p.name)
        st.image(str(p), use_container_width=True)


def render_run_browser_tab() -> None:
    st.header("Run Browser")
    st.caption(f"Runs directory: `{RUNS_DIR}` (same convention as `RunService.runs_dir`).")

    run_dirs = _list_run_directories()
    if not run_dirs:
        if not RUNS_DIR.is_dir():
            st.warning(
                "No `runs/` directory found. Create runs from the CLI (e.g. `hexnet train`) or run from the repo root."
            )
        else:
            st.warning("`runs/` exists but has no subfolders yet.")
        return

    names = [d.name for d in run_dirs]
    if st.session_state.get("run_browser_selected_run") not in names:
        st.session_state.run_browser_selected_run = names[0]

    left, right = st.columns([1, 4])
    with left:
        _render_run_tree(run_dirs)
    with right:
        selected = RUNS_DIR / st.session_state.run_browser_selected_run
        json_files = _json_artifacts(selected)
        plot_paths = _training_plot_paths(selected)

        if plot_paths:
            json_col, plot_col = st.columns([2, 3])
            with json_col:
                _render_json_viewer_section(selected, json_files)
            with plot_col:
                _render_training_plots(plot_paths)
        else:
            _render_json_viewer_section(selected, json_files)
