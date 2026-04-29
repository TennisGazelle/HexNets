import streamlit as st

from hexnets_web.glossary_types import GlossaryNode
from hexnets_web.pages.base_page import BasePage
from hexnets_web.pages.glossary.glossary_data import GLOSSARY_ROOT


def render_glossary_node(
    node: GlossaryNode, query: str, as_expander: bool = True
) -> None:
    if query and query not in node.search_blob:
        return

    def render_content():
        st.markdown(node.english)
        if node.tags:
            st.caption(" · ".join(node.tags))
        if node.math_latex:
            st.latex(node.math_latex)
        if node.good_for:
            st.markdown(f"**Good for:** {node.good_for}")
        if node.example:
            st.markdown(f"**Example:** {node.example}")

    if as_expander:
        with st.expander(node.title):
            render_content()
            for c in node.children:
                render_glossary_node(c, query)
    else:
        st.markdown(f"#### {node.title}")
        render_content()
        if node.children:
            st.markdown("#### See also" + " · ".join([c.title for c in node.children]))


class GlossaryPage(BasePage):
    def render(self) -> None:
        st.header("Glossary")
        st.caption(
            "Search matches titles, definitions, formulas, examples, tags, good-for blurbs, "
            "and nested terms. Parents stay visible when a child matches."
        )
        q = st.text_input(
            "Search glossary",
            key="glossary_query",
            placeholder="e.g. identity, R², rotation…",
        )
        query = (q or "").strip().lower()

        filtered = [n for n in GLOSSARY_ROOT if not query or query in n.search_blob]
        if not filtered:
            st.info("No entries match your search.")
            return

        mid = (len(filtered) + 1) // 2
        left, right = filtered[:mid], filtered[mid:]
        col1, col2 = st.columns(2)
        with col1:
            for n in left:
                render_glossary_node(n, query)
        with col2:
            for n in right:
                render_glossary_node(n, query)
