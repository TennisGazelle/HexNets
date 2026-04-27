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


def _render_run_tree(run_dirs: list[pathlib.Path]) -> None:
    st.subheader("Runs tree (read-only)")
    for d in run_dirs:
        with st.expander(d.name, expanded=False):
            entries = sorted(d.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
            if not entries:
                st.caption("(empty)")
                continue
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


def render_run_browser_tab() -> None:
    st.header("Run Browser")
    st.caption(f"Runs directory: `{RUNS_DIR}` (same convention as `RunService.runs_dir`).")

    run_dirs = _list_run_directories()
    if not run_dirs:
        if not RUNS_DIR.is_dir():
            st.warning("No `runs/` directory found. Create runs from the CLI (e.g. `hexnet train`) or run from the repo root.")
        else:
            st.warning("`runs/` exists but has no subfolders yet.")
        return

    names = [d.name for d in run_dirs]
    if st.session_state.get("run_browser_selected_run") not in names:
        st.session_state.run_browser_selected_run = names[0]

    st.selectbox("Selected run (for JSON viewer)", options=names, key="run_browser_selected_run")

    selected = RUNS_DIR / st.session_state.run_browser_selected_run
    json_files = _json_artifacts(selected)

    st.subheader("JSON viewer")
    if not json_files:
        st.info("No `.json` files in this run folder root.")
    else:
        if st.session_state.get("run_browser_selected_json") not in json_files:
            st.session_state.run_browser_selected_json = json_files[0]
        st.selectbox("JSON file", options=json_files, key="run_browser_selected_json")
        _render_json_viewer(selected, st.session_state.run_browser_selected_json)

    st.markdown("---")
    _render_run_tree(run_dirs)
