import pathlib
from textwrap import dedent

import matplotlib.pyplot as plt
import streamlit as st

from data.dataset import list_registered_dataset_display_names
from networks.activation.activations import get_available_activation_functions
from networks.learning_rate.learning_rate import get_available_learning_rates
from networks.loss.loss import get_available_loss_functions
from hexnets_web.figures import create_matplotlib_figure
from hexnets_web.pages.base_page import BasePage
from hexnets_web.session import update_network
from networks.HexagonalNetwork import HexagonalNeuralNetwork

_DATASET_SAMPLE_OPTIONS = (10, 50, 100, 250, 500, 1000)


class NetworkExplorerPage(BasePage):
    def __init__(self, streamlit_dir: pathlib.Path) -> None:
        self._streamlit_dir = streamlit_dir

    @staticmethod
    def _indices_preview(layer: list, max_len: int = 80) -> str:
        s = repr(layer)
        if len(s) <= max_len:
            return s
        return s[: max_len - 1] + "…"

    def _render_network_info_panel(self) -> None:
        update_network()
        try:
            net = st.session_state.net
            layers = net.dir_W[st.session_state.r]["indices"]
            total_nodes = sum(len(layer) for layer in layers)
            sizes = [len(layer) for layer in layers]
            r_key = st.session_state.r
            total_edges = HexagonalNeuralNetwork.get_parameter_count(st.session_state.n)

            with st.container(border=True):
                st.markdown("##### Network structure")
                m1, m2, m3 = st.columns(3)
                with m1:
                    st.metric("Total nodes", f"{total_nodes:,}")
                with m2:
                    st.metric("Layers", len(layers))
                with m3:
                    st.metric("Total Edges", total_edges)

                st.caption(
                    f"Hex order **n** = `{st.session_state.n}` · layer sizes: " f"`{' → '.join(str(s) for s in sizes)}`"
                )

                # st.markdown("##### Hyperparameters")
                # st.markdown(
                #     "| | |\n"
                #     "|:--|:--|\n"
                #     f"| Activation | `{st.session_state.activation}` |\n"
                #     f"| Loss | `{st.session_state.loss}` |\n"
                #     f"| Learning rate | `{st.session_state.learning_rate}` |\n"
                #     f"| Dataset | `{st.session_state.dataset_type}` |\n"
                #     f"| Samples | `{st.session_state.dataset_num_samples}` |\n"
                # )

                st.markdown("##### Layers (indices per layer)")
                for i, layer in enumerate(layers):
                    with st.container():
                        st.markdown(f"**Layer {i}** · `{len(layer)}` nodes")
                        st.code(self._indices_preview(layer), language=None)
        except Exception as e:
            st.error(f"Error displaying network info: {e}")

    def render(self) -> None:
        st.markdown("---")
        st.markdown("### About")
        st.markdown(dedent("""
                **HexNets** studies **hexagonal neural networks**: models whose connectivity follows a
                hex-layer layout (controlled by order **n**) and optional **rotation** **r** of how
                weights are organized. The goal is to make that geometry **inspectable and reproducible**
                — same parameters in the CLI, in saved runs, and here — so you can reason about structure,
                size, and activations without digging through code on every change.

                **This page** is a lightweight **parameter-driven dashboard**. Adjust the hex-specific
                settings (**n**, **r**) plus activation, loss, learning-rate schedule, and dataset choices;
                the right-hand panel **recomputes the live network** and shows **derived statistics** (total
                nodes, layer sizes, per-layer index lists, edge counts, and similar summaries). It does
                **not** run training loops here — it is for **reading off** what a given configuration
                implies. Use **Generate Graphs** when you want the usual **structure** and **multi-activation**
                figures for the current `(n, r)` and hyperparameters.
                """).strip())

        streamlit_dir = self._streamlit_dir
        st.header("Parameters")

        activation_opts = sorted(get_available_activation_functions())
        if st.session_state.activation not in activation_opts:
            st.session_state.activation = activation_opts[0]
        loss_opts = sorted(get_available_loss_functions())
        if st.session_state.loss not in loss_opts:
            st.session_state.loss = loss_opts[0]
        lr_opts = sorted(get_available_learning_rates())
        if st.session_state.learning_rate not in lr_opts:
            st.session_state.learning_rate = "constant" if "constant" in lr_opts else lr_opts[0]

        col_sliders_and_network, col_info = st.columns(2, gap="large")

        with col_sliders_and_network:
            st.markdown("**Geometry & data**")
            st.session_state.n = st.number_input(
                "Number of nodes (n)",
                min_value=2,
                max_value=25,
                value=st.session_state.n,
                step=1,
                format="%d",
            )

            st.session_state.r = st.slider(
                "Rotation (r)",
                min_value=0,
                max_value=5,
                value=st.session_state.r,
                step=1,
            )

            if st.button("Generate Structure Graph", type="primary"):
                update_network()
                with st.spinner(f"Generating for n={st.session_state.n}, r={st.session_state.r}..."):
                    net = st.session_state.net

                    _filename, fig = net.graph_structure(output_dir=streamlit_dir, medium="matplotlib")
                    buf = create_matplotlib_figure(fig)
                    st.image(buf, use_container_width=True)
                    plt.close(fig)

            # dataset_names = list_registered_dataset_display_names()
            # if st.session_state.dataset_type not in dataset_names:
            #     st.session_state.dataset_type = "identity"
            # st.selectbox("Dataset type", options=dataset_names, key="dataset_type")
            # if st.session_state.dataset_num_samples not in _DATASET_SAMPLE_OPTIONS:
            #     st.session_state.dataset_num_samples = 100
            # st.selectbox(
            #     "Number of data samples",
            #     options=list(_DATASET_SAMPLE_OPTIONS),
            #     key="dataset_num_samples",
            # )

            # st.selectbox("Activation", options=activation_opts, key="activation")
            # st.selectbox("Loss", options=loss_opts, key="loss")
            # st.selectbox(
            #     "Learning rate",
            #     options=lr_opts,
            #     key="learning_rate",
            #     help="Schedule name passed to the live network (same registry as CLI).",
            # )

        with col_info:
            self._render_network_info_panel()
