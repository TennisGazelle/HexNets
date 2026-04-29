"""
Streamlit tab: build and preview `hexnet ...` CLI strings from CliNode metadata.
"""

from __future__ import annotations

import argparse
import shlex
from typing import Any, Iterable

import streamlit as st

from hexnets_web.cli_data import CLI_ROOT
from hexnets_web.cli_types import CliArgNode, CliNode


def _widget_key(cmd_name: str, dest: str) -> str:
    return f"cli_builder__{cmd_name}__{dest}"


def _help_kw(arg: CliArgNode) -> dict[str, str]:
    if arg.help:
        return {"help": arg.help}
    return {}


def _default_index(choices: tuple[Any, ...], default: Any) -> int:
    if default in choices:
        return choices.index(default)
    return 0


def _coerce_value(arg: CliArgNode, raw: Any) -> Any:
    if arg.is_flag:
        return bool(raw)

    if arg.type_kind == "int":
        if raw == "" or raw is None:
            return None
        if isinstance(raw, bool):
            return int(raw)
        if isinstance(raw, (int, float)):
            return int(raw)
        text = str(raw).strip()
        if not text:
            return None
        return int(text)

    if arg.type_kind == "float":
        if raw == "" or raw is None:
            return None
        if isinstance(raw, (int, float)):
            return float(raw)
        text = str(raw).strip()
        if not text:
            return None
        return float(text)

    if arg.type_kind == "path":
        text = str(raw).strip() if raw is not None else ""
        return text if text else None

    text = str(raw).strip() if raw is not None else ""
    return text


def _values_equal(a: Any, b: Any) -> bool:
    if a is None and b is None:
        return True
    if type(a) is type(b) or (isinstance(a, (int, float)) and isinstance(b, (int, float))):
        try:
            return a == b
        except Exception:
            return False
    try:
        return str(a) == str(b)
    except Exception:
        return False


def _format_cli_part(arg: CliArgNode, value: Any) -> list[str]:
    if arg.is_flag:
        if bool(value):
            long_opts = [s for s in arg.option_strings if s.startswith("--")]
            flag = long_opts[0] if long_opts else arg.option_strings[0]
            return [flag]
        return []

    if arg.is_positional:
        text = str(value).strip() if value is not None else ""
        if not text:
            return []
        return [shlex.quote(text)]

    long_opts = [s for s in arg.option_strings if s.startswith("--")]
    short_opts = [s for s in arg.option_strings if s.startswith("-") and not s.startswith("--")]
    opt = long_opts[0] if long_opts else (short_opts[0] if short_opts else arg.dest)

    if arg.type_kind == "bool":
        return [opt, "true" if value else "false"]

    token = shlex.quote(str(value))
    return [opt, token]


GLOBAL_GROUP = "global"


def _render_arg_widgets(args: Iterable[CliArgNode], cmd_name: str) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for arg in args:
        key = _widget_key(cmd_name, arg.dest)
        label = arg.primary_label
        if arg.required:
            label = f"{label} *"

        h = _help_kw(arg)

        if arg.is_flag:
            default_on = bool(arg.default) if arg.default not in (None, argparse.SUPPRESS) else False
            values[arg.dest] = st.checkbox(label, value=default_on, key=key, **h)
            continue

        if arg.choices is not None:
            if arg.default is None:
                omit_label = "— omit —"
                opts = [omit_label, *arg.choices]
                sel = st.selectbox(label, options=opts, key=key, **h)
                values[arg.dest] = None if sel == omit_label else sel
            else:
                idx = _default_index(arg.choices, arg.default)
                values[arg.dest] = st.selectbox(
                    label, options=list(arg.choices), index=idx, key=key, **h
                )
            continue

        if arg.type_kind == "int":
            if arg.default is None:
                values[arg.dest] = st.text_input(
                    label, value="", placeholder="optional", key=key, **h
                )
            else:
                values[arg.dest] = st.number_input(
                    label,
                    value=int(arg.default),
                    step=1,
                    format="%d",
                    key=key,
                    **h,
                )
            continue

        if arg.type_kind == "float":
            if arg.default is None:
                values[arg.dest] = st.text_input(
                    label, value="", placeholder="optional", key=key, **h
                )
            else:
                values[arg.dest] = st.number_input(
                    label,
                    value=float(arg.default),
                    step=0.01,
                    format="%g",
                    key=key,
                    **h,
                )
            continue

        default_text = ""
        if arg.default is not None and arg.default is not argparse.SUPPRESS:
            default_text = str(arg.default)
        values[arg.dest] = st.text_input(label, value=default_text, key=key, **h)

    return values


def _build_preview(cmd: CliNode, values: dict[str, Any]) -> str:
    parts: list[str] = []
    for arg in cmd.args:
        raw = values.get(arg.dest)
        coerced = _coerce_value(arg, raw)

        if arg.is_positional:
            parts.extend(_format_cli_part(arg, coerced))
            continue

        if arg.is_flag:
            parts.extend(_format_cli_part(arg, coerced))
            continue

        if _values_equal(coerced, arg.default):
            continue

        parts.extend(_format_cli_part(arg, coerced))

    return " ".join(["hexnet", cmd.name, *parts])


def render_cli_builder_tab() -> None:
    st.header("CLI Builder")
    st.caption(
        "Pick a subcommand and options; the preview updates as you change controls. "
        "Copy the command from the code block (hover → copy)."
    )

    if not CLI_ROOT.children:
        st.warning("No CLI commands registered.")
        return

    col1, col2, col3, col4 = st.columns([1, 5, 5, 5], vertical_alignment="center")

    with col1:
        st.markdown("### hexnet")

    with col2:
        cmd_labels = [f"{c.name} — {c.help}" for c in CLI_ROOT.children]
        cmd_by_label = dict(zip(cmd_labels, CLI_ROOT.children, strict=True))
        selected_label = st.selectbox("Command", options=cmd_labels, key="cli_builder_command_pick")
        cmd = cmd_by_label[selected_label]

    command_args = [a for a in cmd.args if a.group != GLOBAL_GROUP]
    global_args = [a for a in cmd.args if a.group == GLOBAL_GROUP]

    with col3:
        st.caption("Command options")
        if command_args:
            values_cmd = _render_arg_widgets(command_args, cmd.name)
        else:
            st.caption("_No command-specific options for this subcommand._")
            values_cmd = {}

    with col4:
        st.caption("Global options")
        if global_args:
            values_global = _render_arg_widgets(global_args, cmd.name)
        else:
            st.caption("_No global options for this subcommand._")
            values_global = {}

    values = {**values_cmd, **values_global}
    preview = _build_preview(cmd, values)

    st.markdown("---")
    st.subheader("Preview")
    st.code(preview, language="bash")

    st.button(
        "Run (coming soon)",
        disabled=True,
        help="Execution from the UI is not yet implemented",
        key="cli_builder_run_placeholder",
    )
