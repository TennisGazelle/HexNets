import streamlit as st

from streamlit_app.references import load_multi_activation_image, load_reference_image


def show_rotation_comparison(n=3):
    """Display the rotation comparison table similar to ROTATION_SYSTEM.md"""
    st.header("Rotation Comparison Table")
    st.markdown(
        f"Showing all rotations for **n={n}**. This table recreates the visualization from ROTATION_SYSTEM.md documentation."
    )

    rotation_tabs = st.tabs([f"Rotation {r}" for r in range(6)])

    for r, tab in enumerate(rotation_tabs):
        with tab:
            st.subheader(f"Rotation {r}")

            structure_img = load_reference_image(n, r, "structure")
            activation_img = load_reference_image(n, r, "activation")
            weight_img = load_reference_image(n, r, "weight")
            multi_activation_img = load_multi_activation_image(n)

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.markdown("**Physical Structure**")
                if structure_img:
                    st.image(structure_img, use_container_width=True)
                    st.markdown("The physical structure of the network.")
                else:
                    st.warning(f"Structure image not found for n={n}, r={r}")

            with col2:
                st.markdown("**Activation Pattern**")
                if activation_img:
                    st.image(activation_img, use_container_width=True)
                    st.markdown("The activation pattern of the network, via bitmap")
                else:
                    st.warning(f"Activation image not found for n={n}, r={r}")

            with col3:
                st.markdown("**Weight Matrix**")
                if weight_img:
                    st.image(weight_img, use_container_width=True)
                    st.markdown("The weight matrix of an untrained network.")
                else:
                    st.warning(f"Weight image not found for n={n}, r={r}")

            with col4:
                st.markdown("**Multi-Activation Overlay**")
                if multi_activation_img:
                    st.image(multi_activation_img, use_container_width=True)
                    st.markdown(
                        "This view shows all 6 rotations overlaid on a single matrix, with each rotation shown in a different color."
                    )
                else:
                    st.warning(f"Multi-activation image not found for n={n}. Generate it with: `hexnet ref --all`")


def render_rotation_comparison_tab() -> None:
    n_selector = st.selectbox(
        "Select n (dimension) for rotation comparison",
        options=[2, 3, 4, 5, 6, 7, 8],
        index=1,
        key="rotation_comparison_n",
    )
    show_rotation_comparison(n=n_selector)
