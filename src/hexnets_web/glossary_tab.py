import streamlit as st

from hexnets_web.glossary_data import GLOSSARY_ROOT, GlossaryNode


def render_glossary_node(node: GlossaryNode, query: str) -> None:
    if query and query not in node.search_blob:
        return
    with st.expander(node.title):
        st.markdown(node.english)
        if node.tags:
            st.caption(" · ".join(node.tags))
        if node.math_latex:
            st.latex(node.math_latex)
        if node.good_for:
            st.markdown(f"**Good for:** {node.good_for}")
        if node.example:
            st.markdown(f"**Example:** {node.example}")
        for c in node.children:
            render_glossary_node(c, query)


def render_glossary_tab() -> None:
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
