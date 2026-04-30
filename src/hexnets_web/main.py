import streamlit as st

from hexnets_web.session import initialize_session_state


def run() -> None:
    st.set_page_config(page_title="HexNet Visualizer", page_icon=None, layout="wide")

    initialize_session_state()

    st.title("Hexagonal Neural Network Visualizer")

    pages = [
        st.Page(
            "hexnets_web/pages/cli/cli_page.py",
            title="CLI Builder",
            default=True,
            icon=":material/terminal:",
        ),
        st.Page(
            "hexnets_web/pages/network_explorer/network_explorer_page.py",
            title="Network Explorer",
            icon=":material/hub:",
        ),
        st.Page(
            "hexnets_web/pages/rotation_comparison/rotation_comparison_page.py",
            title="Rotation Comparison",
            icon=":material/compare:",
        ),
        st.Page(
            "hexnets_web/pages/lesion_lab/lesion_lab_page.py",
            title="Lesion Lab",
            icon=":material/science:",
        ),
        st.Page(
            "hexnets_web/pages/run_browser/run_browser_page.py",
            title="Run Browser",
            icon=":material/folder_open:",
        ),
        st.Page(
            "hexnets_web/pages/glossary/glossary_page.py",
            title="Glossary",
            icon=":material/menu_book:",
        ),
    ]

    pg = st.navigation(pages, position="sidebar")
    pg.run()
