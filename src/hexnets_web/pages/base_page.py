"""Abstract base for Streamlit multipage routes."""

from __future__ import annotations

from abc import ABC, abstractmethod


class BasePage(ABC):
    """Each sidebar page implements ``render()`` with its Streamlit UI."""

    @abstractmethod
    def render(self) -> None:
        """Draw this page (widgets, charts, etc.)."""
