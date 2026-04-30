import streamlit as st

from hexnets_web.pages.base_page import BasePage
from hexnets_web.references import load_multi_activation_image, load_reference_image


def _show_multi_activation_column(n: int) -> None:
    multi_activation_img = load_multi_activation_image(n)
    st.markdown("**Multi-Activation Overlay**")
    st.caption("Per **n** only (same for every rotation).")
    if multi_activation_img:
        st.image(multi_activation_img, use_container_width=True)
        st.markdown("All six rotations overlaid on one matrix; each rotation in a different color.")
    else:
        st.warning(f"Multi-activation image not found for n={n}. Generate with: `hexnet ref --all`")


def _show_three_reference_images(n: int, r: int) -> None:
    structure_img = load_reference_image(n, r, "structure")
    activation_img = load_reference_image(n, r, "activation")
    weight_img = load_reference_image(n, r, "weight")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("**Physical Structure**")
        if structure_img:
            st.image(structure_img, use_container_width=True)
            st.markdown("The physical structure of the network.")
        else:
            st.warning(f"Structure image not found for n={n}, r={r}")

    with c2:
        st.markdown("**Activation Pattern**")
        if activation_img:
            st.image(activation_img, use_container_width=True)
            st.markdown("The activation pattern of the network, via bitmap")
        else:
            st.warning(f"Activation image not found for n={n}, r={r}")

    with c3:
        st.markdown("**Weight Matrix**")
        if weight_img:
            st.image(weight_img, use_container_width=True)
            st.markdown("The weight matrix of an untrained network.")
        else:
            st.warning(f"Weight image not found for n={n}, r={r}")


class RotationComparisonPage(BasePage):
    def render(self) -> None:
        st.header("Rotation Comparison")
        st.markdown(
            "Browse pre-generated reference images by **n** and **r**. "
            "Sliders and multi-activation are on the left; structure, activation, and weight for the chosen rotation are on the right. "
            "These controls use separate session keys from Network Explorer so they do not change the live network."
        )

        n = st.session_state.rotation_comparison_n
        r = st.session_state.rotation_comparison_r

        left, right = st.columns([1, 3])
        with left:
            st.slider(
                "Number of nodes (n)",
                min_value=2,
                max_value=8,
                step=1,
                key="rotation_comparison_n",
            )
            n = st.session_state.rotation_comparison_n
            _show_multi_activation_column(n)

        with right:
            st.slider(
                "Rotation (r)",
                min_value=0,
                max_value=5,
                step=1,
                key="rotation_comparison_r",
            )
            n = st.session_state.rotation_comparison_n
            r = st.session_state.rotation_comparison_r
            _show_three_reference_images(n, r)
