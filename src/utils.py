import json
import pathlib
from typing import Any, List
from tabulate import tabulate


def table_print(headers: List[str], data: List[List]):
    print(tabulate(data, headers, tablefmt="grid"))


def read_json_from_path(fileref: pathlib.Path, description: str) -> Any:
    """Load JSON from ``fileref`` with actionable errors for run ingestion / tooling."""
    abs_path = fileref.resolve()
    if not fileref.exists():
        raise ValueError(
            f"Cannot load {description}: expected file does not exist:\n  {abs_path}\n"
            "Repair or replace the run directory."
        )
    try:
        with open(fileref, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Cannot load {description}: file is not valid JSON:\n  {abs_path}\n"
            f"  line {e.lineno}, column {e.colno}: {e.msg}\n"
            "Repair or replace the run directory."
        ) from e
    except UnicodeDecodeError as e:
        raise ValueError(f"Cannot load {description}: file is not valid UTF-8:\n  {abs_path}\n  {e}") from e


def read_json_object(fileref: pathlib.Path, description: str) -> dict:
    """Like :func:`read_json_from_path` but requires a JSON object (dict)."""
    data = read_json_from_path(fileref, description)
    if not isinstance(data, dict):
        raise ValueError(
            f"Cannot load {description}: expected a JSON object in\n  {fileref.resolve()}\n"
            f"got {type(data).__name__}. This run may be from an incompatible tool version or a partial export."
        )
    return data


def get_json_file_contents(fileref: pathlib.Path) -> dict:
    """Deprecated for run files; prefer :func:`read_json_object`. Kept for callers expecting dict-only JSON."""
    return read_json_object(fileref, fileref.name)


class Colors:
    """Terminal color codes for colored output."""

    RED = "\033[0;31m"
    GREEN = "\033[0;32m"
    YELLOW = "\033[1;33m"
    BLUE = "\033[0;34m"
    NC = "\033[0m"  # No Color
