"""Streamlit route: Network Explorer."""

import pathlib

from hexnets_web.pages.network_explorer.network_explorer import NetworkExplorerPage

NetworkExplorerPage(pathlib.Path("./reference").resolve()).render()
