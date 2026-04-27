import pathlib

import matplotlib.pyplot as plt
import streamlit as st

from commands.command import get_dataset
from networks.activation.activations import get_available_activation_functions
from networks.loss.loss import get_available_loss_functions
from streamlit_app.figures import create_matplotlib_figure
from streamlit_app.metrics_explainer import render_training_metrics_expander
from streamlit_app.session import update_network


def render_network_explorer_tab(streamlit_dir: pathlib.Path) -> None:
    st.header("Parameters")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.session_state.n = st.slider(
            "Number of nodes (n)", min_value=2, max_value=8, value=st.session_state.n, step=1
        )

        st.session_state.r = st.slider("Rotation (r)", min_value=0, max_value=5, value=st.session_state.r, step=1)

        st.session_state.activation = st.selectbox("Activation", get_available_activation_functions(), index=0)

        st.session_state.loss = st.selectbox("Loss", get_available_loss_functions(), index=0)

        if st.button("Generate Graphs"):
            update_network()
            with st.spinner(f"Generating for n={st.session_state.n}, r={st.session_state.r}..."):
                net = st.session_state.net

                col_structure, col_multi_activation = st.columns(2)

                with col_structure:
                    filename, fig = net.graph_structure(output_dir=streamlit_dir, medium="matplotlib")
                    buf = create_matplotlib_figure(fig)
                    st.image(buf, use_container_width=True)
                    plt.close(fig)

                with col_multi_activation:
                    filename, fig = net._graph_multi_activation(detail="", output_dir=streamlit_dir)
                    buf = create_matplotlib_figure(fig)
                    st.image(buf, use_container_width=True)
                    plt.close(fig)

        if st.button("Train Network"):
            update_network()
            with st.spinner("Training on identity dataset"):
                data = get_dataset(st.session_state.n, 100, type="identity")
                net = st.session_state.net
                loss, reg_score, r2, fig = net.train_animated(data, epochs=10, pause=0, output_dir=streamlit_dir)
                adj = net.training_metrics[net.r].adjusted_r_squared[-1]
                st.session_state.last_metrics = {
                    "loss": float(loss),
                    "regression_score": float(reg_score),
                    "r_squared": float(r2),
                    "adjusted_r_squared": float(adj),
                }
                buf = create_matplotlib_figure(fig)
                st.image(buf, use_container_width=True)
                plt.close(fig)

    with col2:
        try:
            net = st.session_state.net
            layers = net.dir_W[st.session_state.r]["indices"]

            st.subheader("Network Information")
            st.write(f"**Total nodes:** {sum(len(layer) for layer in layers)}")
            st.write(f"**Number of layers:** {len(layers)}")
            st.write(f"**Layer sizes:** {[len(layer) for layer in layers]}")
            st.write(f"**n (nodes):** {st.session_state.n}")
            st.write(f"**r (rotation):** {st.session_state.r}")
            st.write(f"**Activation:** {st.session_state.activation}")
            st.write(f"**Loss:** {st.session_state.loss}")

            with st.expander("Layer Indices"):
                for i, layer in enumerate(layers):
                    st.write(f"Layer {i}: {layer}")
        except Exception as e:
            st.error(f"Error displaying network info: {e}")

    render_training_metrics_expander()

    st.markdown("---")
    st.markdown("### About")
    st.write("HexNets is a tool for working with hexagonal neural networks.")
