"""
CLI builder tree types (stdlib only; safe for import from commands.command).

Maps argparse parsers to frozen CliNode / CliArgNode records for Streamlit.
"""

from __future__ import annotations

import argparse
import pathlib
from argparse import ArgumentParser
from dataclasses import dataclass
from typing import Any, Literal

CliTypeKind = Literal["bool", "int", "float", "str", "path"]

# argparse default group titles vary by version ("optional arguments" vs "options")
_GROUP_TITLES_AS_COMMAND = frozenset({"positional arguments", "options", "optional arguments"})


def _resolve_group_title(raw: str | None) -> str:
    if not raw or raw in _GROUP_TITLES_AS_COMMAND:
        return "command"
    return raw


def _action_group_lookup(parser: ArgumentParser) -> dict[int, str]:
    """Map action object id -> normalized group name (for CliArgNode.group)."""
    mapping: dict[int, str] = {}
    for grp in parser._action_groups:
        title = _resolve_group_title(grp.title)
        for action in grp._group_actions:
            mapping[id(action)] = title
    return mapping


def _tuple_or_none(seq: Any) -> tuple[Any, ...] | None:
    if seq is None:
        return None
    return tuple(seq)


def _infer_type_kind(action: argparse.Action) -> CliTypeKind:
    atype = action.type
    if atype is None:
        return "str"
    if atype is bool:
        return "bool"
    if atype is int:
        return "int"
    if atype is float:
        return "float"
    try:
        if isinstance(atype, type) and issubclass(atype, pathlib.Path):
            return "path"
    except TypeError:
        pass
    return "str"


def _is_store_true(action: argparse.Action) -> bool:
    return type(action).__name__ == "_StoreTrueAction"


def _is_store_false(action: argparse.Action) -> bool:
    return type(action).__name__ == "_StoreFalseAction"


def _should_skip_action(action: argparse.Action) -> bool:
    kind = type(action).__name__
    if kind in ("HelpAction", "_HelpAction"):
        return True
    if kind == "_SubParsersAction":
        return True
    return False


def _flag_meta(action: argparse.Action) -> tuple[bool, bool]:
    """Returns (is_flag, flag_value_when_set)."""
    if _is_store_true(action):
        return True, True
    if _is_store_false(action):
        return True, False
    return False, True


def cli_node_from_parser(*, name: str, help_text: str, parser: ArgumentParser) -> CliNode:
    """Build a CliNode from a configured ArgumentParser (flat options only)."""
    group_by_action_id = _action_group_lookup(parser)
    args: list[CliArgNode] = []
    for action in parser._actions:
        if _should_skip_action(action):
            continue

        option_strings = tuple(action.option_strings or ())
        is_flag, flag_when_set = _flag_meta(action)

        if is_flag:
            type_kind: CliTypeKind = "bool"
            choices = None
        else:
            type_kind = _infer_type_kind(action)
            choices = _tuple_or_none(action.choices)

        group = group_by_action_id.get(id(action), "command")

        args.append(
            CliArgNode(
                dest=action.dest,
                option_strings=option_strings,
                help=action.help,
                default=action.default,
                choices=choices,
                type_kind=type_kind,
                required=bool(action.required),
                is_flag=is_flag,
                flag_value_when_set=flag_when_set,
                group=group,
            )
        )

    return CliNode(name=name, help=help_text, args=tuple(args))


@dataclass(frozen=True)
class CliArgNode:
    dest: str
    option_strings: tuple[str, ...]
    help: str | None
    default: Any
    choices: tuple[Any, ...] | None
    type_kind: CliTypeKind
    required: bool
    is_flag: bool
    flag_value_when_set: bool
    group: str = "command"

    @property
    def is_positional(self) -> bool:
        return len(self.option_strings) == 0

    @property
    def primary_label(self) -> str:
        long_opts = [s for s in self.option_strings if s.startswith("--")]
        if long_opts:
            return long_opts[0]
        short_opts = [s for s in self.option_strings if s.startswith("-") and not s.startswith("--")]
        if short_opts:
            return short_opts[0]
        return self.dest


@dataclass(frozen=True)
class CliNode:
    name: str
    help: str
    args: tuple[CliArgNode, ...] = ()
    children: tuple["CliNode", ...] = ()
