"""
Glossary tree node type (stdlib only; safe for CLI imports via data.dataset).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class GlossaryNode:
    title: str
    english: str
    aliases: tuple[str, ...] = ()
    math_latex: str | None = None
    example: str | None = None
    good_for: str | None = None
    tags: tuple[str, ...] = ()
    children: tuple["GlossaryNode", ...] = ()
    search_blob: str = field(default="", repr=False)
