import pathlib

import streamlit as st

from streamlit_app.glossary_tab import render_glossary_tab
from streamlit_app.lesion_lab import render_lesion_lab_tab
from streamlit_app.network_explorer import render_network_explorer_tab
from streamlit_app.rotation_comparison import render_rotation_comparison_tab
from streamlit_app.run_browser import render_run_browser_tab
from streamlit_app.session import initialize_session_state


def run() -> None:
    streamlit_dir = pathlib.Path("./reference").resolve()
    st.set_page_config(page_title="HexNet Visualizer", page_icon=None, layout="wide")

    initialize_session_state()

    st.title("HexNet Visualizer")
    st.subheader("Hexagonal Neural Network Visualizer")

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["Network Explorer", "Rotation Comparison", "Lesion Lab", "Run Browser", "Glossary"]
    )

    with tab1:
        render_network_explorer_tab(streamlit_dir)
    with tab2:
        render_rotation_comparison_tab()
    with tab3:
        render_lesion_lab_tab()
    with tab4:
        render_run_browser_tab()
    with tab5:
        render_glossary_tab()
