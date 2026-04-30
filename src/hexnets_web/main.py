import streamlit as st
import streamlit.components.v1 as components

from hexnets_web.session import initialize_session_state

# Paste your full Buy Me a Coffee snippet (the whole <script ...></script> block).
# Use components.html here — Streamlit strips scripts from st.markdown even with
# unsafe_allow_html=True. Height should clear the button (~50–70).
_BUY_ME_A_COFFEE_HTML = "<script type='text/javascript' src='https://cdnjs.buymeacoffee.com/1.0.0/button.prod.min.js' data-name='bmc-button' data-slug='tennisgazelle' data-color='#BD5FFF' data-emoji='☕'  data-font='Lato' data-text='Buy me a coffee' data-outline-color='#000000' data-font-color='#ffffff' data-coffee-color='#FFDD00' ></script>"


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

    height = 80
    if _BUY_ME_A_COFFEE_HTML.strip():
        with st.sidebar:
            components.html(_BUY_ME_A_COFFEE_HTML, height=height)
