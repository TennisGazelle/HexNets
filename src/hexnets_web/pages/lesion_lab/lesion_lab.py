import streamlit as st

from hexnets_web.pages.base_page import BasePage


class LesionLabPage(BasePage):
    def render(self) -> None:
        st.header("Lesion Lab")
        st.info("Coming soon — still running experiments.")
