import hashlib
import json
import pathlib
import random

import streamlit as st

from hexnets_web.pages.base_page import BasePage
from hexnets_web.pages.run_browser.fam_a_signatures import (
    FamASignature,
    build_fam_a_pair_index,
    dropdown_options_for_field,
    resolve_picker_signature,
)
from hexnets_web.pages.run_browser.run_validation import (
    discover_valid_runs_under,
    is_valid_run_dir,
    missing_run_artifacts,
)
from hexnets_web.pages.run_browser.training_plots import _training_plot_paths

# Align with RunService.runs_dir (process cwd, typically repo root).
RUNS_DIR = pathlib.Path("runs/").resolve()
FAM_A_ROOT = RUNS_DIR / "e2etest-famA"

_EMPTY_SELECT = "(empty)"
_NUM_COMPARE_SLOTS = 2


def _init_run_browser_session_state() -> None:
    if "run_browser_session_paths" not in st.session_state:
        st.session_state.run_browser_session_paths = []
    for legacy_key in ("run_browser_slot_2", "run_browser_slot_select_2"):
        st.session_state.pop(legacy_key, None)
    for i in range(_NUM_COMPARE_SLOTS):
        key = f"run_browser_slot_{i}"
        if key not in st.session_state:
            st.session_state[key] = None


def _path_widget_key(path: pathlib.Path) -> str:
    return hashlib.sha256(str(path.resolve()).encode()).hexdigest()[:20]


def _assign_run_to_slot(slot: int, path: str) -> None:
    st.session_state[f"run_browser_slot_{slot}"] = path
    sel_key = f"run_browser_slot_select_{slot}"
    st.session_state[sel_key] = path if path else _EMPTY_SELECT


def _clear_slot(slot: int) -> None:
    st.session_state[f"run_browser_slot_{slot}"] = None
    st.session_state[f"run_browser_slot_select_{slot}"] = _EMPTY_SELECT


def _remove_session_path(path: str) -> None:
    st.session_state.run_browser_session_paths = [p for p in st.session_state.run_browser_session_paths if p != path]
    for i in range(_NUM_COMPARE_SLOTS):
        if st.session_state.get(f"run_browser_slot_{i}") == path:
            _clear_slot(i)


def _add_session_path(path: str) -> None:
    if path not in st.session_state.run_browser_session_paths:
        st.session_state.run_browser_session_paths.append(path)


def _try_add_path_from_input() -> None:
    raw = (st.session_state.get("run_browser_add_path_input") or "").strip()
    if not raw:
        st.session_state.run_browser_add_feedback = "Enter a directory path."
        return
    p = pathlib.Path(raw).expanduser()
    try:
        p = p.resolve()
    except OSError:
        st.session_state.run_browser_add_feedback = f"Could not resolve path: {raw!r}"
        return
    if not p.is_dir():
        st.session_state.run_browser_add_feedback = f"Not a directory: {p}"
        return
    miss = missing_run_artifacts(p)
    if miss:
        st.session_state.run_browser_add_feedback = f"Not a valid run (missing {', '.join(miss)}): {p}"
        return
    ps = str(p)
    if ps in st.session_state.run_browser_session_paths:
        st.session_state.run_browser_add_feedback = f"Already in the tree: {p}"
        return
    _add_session_path(ps)
    st.session_state.run_browser_add_feedback = f"Added: {p}"


def _render_json_config(run_path: pathlib.Path) -> None:
    path = run_path / "config.json"
    if not path.is_file():
        st.warning("Missing `config.json`.")
        return
    raw = path.read_text(encoding="utf-8", errors="replace")
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        st.caption("Could not parse as JSON — showing raw text.")
        st.code(raw, language="json")
        return
    st.json(parsed)


def _render_training_plots(paths: list[pathlib.Path]) -> None:
    st.subheader("Training plots")
    if not paths:
        st.info("No training plot PNGs found (expected `plots/*net_training_*.png` under the run directory).")
        return
    for p in paths:
        st.caption(p.name)
        st.image(str(p), use_container_width=True)


def _class_sort_key(cls: str) -> tuple[int, str]:
    if cls == "hex":
        return (0, cls)
    if cls == "mlp":
        return (1, cls)
    return (2, cls)


def _fam_a_flat_paths(grouped: dict[str, list[pathlib.Path]]) -> list[str]:
    out: list[str] = []
    for paths in grouped.values():
        for p in paths:
            out.append(str(p.resolve()))
    return sorted(set(out))


def _render_run_row(run_p: pathlib.Path, slot_labels: tuple[str, ...]) -> None:
    rp = str(run_p.resolve())
    c0, c1, c2 = st.columns([2.2, 1, 1])
    with c0:
        st.caption(run_p.name)
    for i, col in enumerate((c1, c2)):
        with col:
            st.button(
                slot_labels[i],
                key=f"rb_asg_{_path_widget_key(run_p)}_{i}",
                on_click=_assign_run_to_slot,
                kwargs={"slot": i, "path": rp},
            )


def _render_left_tree(
    fam_grouped: dict[str, list[pathlib.Path]],
) -> None:
    st.subheader("Runs tree")

    st.markdown("**Add run by path**")
    st.text_input(
        "Run directory (absolute or relative)",
        key="run_browser_add_path_input",
        placeholder="e.g. runs/e2etest-smoke/hex-n3-r0",
    )
    st.button("Add run", on_click=_try_add_path_from_input)
    fb = st.session_state.get("run_browser_add_feedback")
    if fb:
        st.caption(fb)

    st.divider()
    st.markdown("**Family A** (`runs/e2etest-famA/`)")
    if not FAM_A_ROOT.is_dir():
        st.caption("Folder not present yet (e.g. run e2e benchmarks to populate).")
    elif not fam_grouped:
        st.caption("No valid run subfolders (need `config.json`, `manifest.json`, `training_metrics.json`).")
    else:
        class_order = sorted(fam_grouped.keys(), key=_class_sort_key)
        for j, cls in enumerate(class_order):
            runs = fam_grouped[cls]
            expanded = j < 2
            with st.expander(f"{cls} ({len(runs)} runs)", expanded=expanded):
                for run_p in runs:
                    _render_run_row(run_p, ("→1", "→2"))

    st.divider()
    st.markdown("**Added this session**")
    sess = st.session_state.run_browser_session_paths
    if not sess:
        st.caption("None yet — use **Add run** above.")
    else:
        for sp in sess:
            p = pathlib.Path(sp)
            st.markdown(f"`{p.name}`")
            r0, r1, r2 = st.columns([2.0, 1, 1])
            with r0:
                st.button(
                    "Remove",
                    key=f"rb_rm_{_path_widget_key(p)}",
                    on_click=_remove_session_path,
                    kwargs={"path": sp},
                )
            for i, col in enumerate((r1, r2)):
                with col:
                    st.button(
                        ("→1", "→2")[i],
                        key=f"rb_sess_asg_{_path_widget_key(p)}_{i}",
                        on_click=_assign_run_to_slot,
                        kwargs={"slot": i, "path": sp},
                    )


def _sync_select_widgets(selectable: list[str]) -> None:
    """Ensure slot selectbox session values stay in the option set."""
    opts = [_EMPTY_SELECT] + selectable
    for i in range(_NUM_COMPARE_SLOTS):
        sk = f"run_browser_slot_select_{i}"
        slot = st.session_state.get(f"run_browser_slot_{i}")
        if sk not in st.session_state:
            st.session_state[sk] = _EMPTY_SELECT if not slot else slot
        val = st.session_state.get(sk)
        if val != _EMPTY_SELECT and val not in selectable:
            st.session_state[sk] = _EMPTY_SELECT
            st.session_state[f"run_browser_slot_{i}"] = None
        elif val == _EMPTY_SELECT:
            st.session_state[f"run_browser_slot_{i}"] = None
        else:
            st.session_state[f"run_browser_slot_{i}"] = val


def _on_slot_select_change(slot: int) -> None:
    key = f"run_browser_slot_select_{slot}"
    val = st.session_state[key]
    st.session_state[f"run_browser_slot_{slot}"] = None if val == _EMPTY_SELECT else val


def _sync_fam_a_picker_from_signature(sig: FamASignature) -> None:
    st.session_state["fam_a_picker_dataset"] = sig.dataset
    st.session_state["fam_a_picker_activation"] = sig.activation
    st.session_state["fam_a_picker_loss"] = sig.loss
    st.session_state["fam_a_picker_learning_rate"] = sig.learning_rate
    st.session_state["fam_a_picker_n_token"] = sig.n_token


def _random_fam_a_click(signatures: list[FamASignature]) -> None:
    if not signatures:
        return
    _sync_fam_a_picker_from_signature(random.choice(signatures))


def _render_fixed_model_column(title: str, run_path: pathlib.Path | None) -> None:
    st.subheader(title)
    if run_path is None:
        st.info("No run on disk for this signature.")
        return
    if not is_valid_run_dir(run_path):
        st.warning("Run folder no longer looks valid on disk.")
        return
    _render_json_config(run_path)
    _render_training_plots(_training_plot_paths(run_path))


def _render_fam_a_picker_tab() -> None:
    st.subheader("Family A picker")
    st.caption(
        "Names follow `scripts/e2e-bench-A.sh`: `model-dataset-activation-loss-learning_rate-n{dims}`. "
        "Columns are **Hex** (left) and **MLP** (right)."
    )

    index = build_fam_a_pair_index(FAM_A_ROOT)
    if not index:
        st.caption(
            "No valid e2e-style runs under `runs/e2etest-famA/` "
            "(expected folders like `hex-<dataset>-<activation>-<loss>-<lr>-n4`)."
        )
        return

    picked = {
        "dataset": st.session_state.get("fam_a_picker_dataset", ""),
        "activation": st.session_state.get("fam_a_picker_activation", ""),
        "loss": st.session_state.get("fam_a_picker_loss", ""),
        "learning_rate": st.session_state.get("fam_a_picker_learning_rate", ""),
        "n_token": st.session_state.get("fam_a_picker_n_token", ""),
    }
    sig = resolve_picker_signature(index, picked)
    _sync_fam_a_picker_from_signature(sig)

    sigs = sorted(index.keys())
    row1 = st.columns([1, 1, 1, 1, 1, 0.72])
    ds_opts = dropdown_options_for_field(index, "dataset")
    with row1[0]:
        st.selectbox("Dataset", options=ds_opts, key="fam_a_picker_dataset")
    d = st.session_state["fam_a_picker_dataset"]
    act_opts = dropdown_options_for_field(index, "activation", dataset=d)
    with row1[1]:
        st.selectbox("Activation", options=act_opts, key="fam_a_picker_activation")
    a = st.session_state["fam_a_picker_activation"]
    loss_opts = dropdown_options_for_field(index, "loss", dataset=d, activation=a)
    with row1[2]:
        st.selectbox("Loss", options=loss_opts, key="fam_a_picker_loss")
    l = st.session_state["fam_a_picker_loss"]
    lr_opts = dropdown_options_for_field(index, "learning_rate", dataset=d, activation=a, loss=l)
    with row1[3]:
        st.selectbox("Learning rate", options=lr_opts, key="fam_a_picker_learning_rate")
    lr = st.session_state["fam_a_picker_learning_rate"]
    n_opts = dropdown_options_for_field(index, "n_token", dataset=d, activation=a, loss=l, learning_rate=lr)
    with row1[4]:
        st.selectbox(
            "Dimensions", options=n_opts, key="fam_a_picker_n_token", help="Suffix like `n4` from the run folder name."
        )
    with row1[5]:
        st.button(
            "Random",
            help="Pick a random signature present for at least one of hex/mlp.",
            on_click=_random_fam_a_click,
            kwargs={"signatures": sigs},
        )

    entry = index[sig]
    hex_p, mlp_p = entry["hex"], entry["mlp"]
    if hex_p is not None and mlp_p is not None:
        st.caption("Disk: full hex + mlp pair for this signature.")
    elif hex_p is not None or mlp_p is not None:
        st.caption("Disk: partial pair — only one of hex/mlp exists for this signature.")
    else:
        st.caption("Disk: no paths (unexpected).")

    hex_col, mlp_col = st.columns(2, gap="medium")
    with hex_col:
        _render_fixed_model_column("Hex", hex_p)
    with mlp_col:
        _render_fixed_model_column("MLP", mlp_p)


def _render_manual_compare_tab(
    fam_grouped: dict[str, list[pathlib.Path]],
    selectable: list[str],
) -> None:
    tree, c1, c2 = st.columns([0.75, 1, 1], gap="medium")
    with tree:
        _render_left_tree(fam_grouped)
    with c1:
        _render_compare_slot(0, selectable)
    with c2:
        _render_compare_slot(1, selectable)


def _render_compare_slot(
    slot: int,
    selectable: list[str],
) -> None:
    st.subheader(f"Compare {slot + 1}")
    opts = [_EMPTY_SELECT] + selectable
    slot_val = st.session_state.get(f"run_browser_slot_{slot}")
    if slot_val is None or slot_val not in selectable:
        ix = 0
    else:
        ix = opts.index(slot_val)

    st.selectbox(
        "Run",
        options=opts,
        index=ix,
        key=f"run_browser_slot_select_{slot}",
        on_change=_on_slot_select_change,
        kwargs={"slot": slot},
    )

    path_s = st.session_state.get(f"run_browser_slot_{slot}")
    if not path_s:
        st.info("Empty — pick a run from the tree or assign above.")
        return

    run_path = pathlib.Path(path_s)
    if not is_valid_run_dir(run_path):
        st.warning("Run folder no longer looks valid on disk.")
        return

    _render_json_config(run_path)
    _render_training_plots(_training_plot_paths(run_path))


class RunBrowserPage(BasePage):
    def render(self) -> None:
        st.header("Run Browser")
        st.caption(f"Runs directory: `{RUNS_DIR}` (same convention as `RunService.runs_dir`).")

        _init_run_browser_session_state()
        fam_grouped = discover_valid_runs_under(FAM_A_ROOT)
        fam_flat = _fam_a_flat_paths(fam_grouped)
        selectable = sorted(set(fam_flat) | set(st.session_state.run_browser_session_paths))
        _sync_select_widgets(selectable)

        tab_picker, tab_manual = st.tabs(["Family A picker", "Manual compare"])
        with tab_picker:
            _render_fam_a_picker_tab()
        with tab_manual:
            _render_manual_compare_tab(fam_grouped, selectable)
