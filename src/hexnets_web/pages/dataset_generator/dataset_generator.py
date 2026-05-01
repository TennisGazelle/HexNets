from __future__ import annotations

import numpy as np
import streamlit as st
from streamlit_echarts import st_echarts

from data.dataset import (
    DATASET_FUNCTIONS,
    InputSamplingMode,
    build_registered_dataset,
    list_registered_dataset_display_names,
)
from hexnets_web.pages.base_page import BasePage
from hexnets_web.pages.glossary.glossary import render_glossary_node


def _scatter_pairs_from_2d(arr: np.ndarray) -> tuple[list[list[float]], str, str]:
    """Scatter points from a 2D array (samples x dims): first two dims, or index vs value if d == 1."""
    if arr.ndim != 2:
        raise ValueError("Expected a 2D array")
    n, d = arr.shape
    if d >= 2:
        pairs = [[float(a), float(b)] for a, b in zip(arr[:, 0], arr[:, 1])]
        return pairs, "Dimension 0", "Dimension 1"
    pairs = [[float(i), float(arr[i, 0])] for i in range(n)]
    return pairs, "Sample index", "Value (dim 0)"


def _echarts_scatter_option(
    *,
    title: str,
    series_data: list[list[float]],
    x_label: str,
    y_label: str,
) -> dict:
    xs = [p[0] for p in series_data]
    ys = [p[1] for p in series_data]
    pad_x = 0.05 * (max(xs) - min(xs) or 1.0)
    pad_y = 0.05 * (max(ys) - min(ys) or 1.0)
    return {
        "animation": False,
        "title": {"text": title, "left": "center", "top": 4},
        "grid": {"left": "12%", "right": "8%", "top": "18%", "bottom": "12%"},
        "tooltip": {"trigger": "item"},
        "xAxis": {
            "name": x_label,
            "nameLocation": "middle",
            "nameGap": 28,
            "min": min(xs) - pad_x,
            "max": max(xs) + pad_x,
        },
        "yAxis": {
            "name": y_label,
            "nameLocation": "middle",
            "nameGap": 36,
            "min": min(ys) - pad_y,
            "max": max(ys) + pad_y,
        },
        "series": [
            {
                "type": "scatter",
                "symbolSize": 6,
                "data": series_data,
            }
        ],
    }


class DatasetGeneratorPage(BasePage):
    def render(self) -> None:
        st.header("Dataset Generator")
        tab_dataset, tab_noise = st.tabs(["Dataset", "Noise"])
        with tab_noise:
            pass
        with tab_dataset:
            self._render_dataset_tab()

    def _render_dataset_tab(self) -> None:
        names = list_registered_dataset_display_names()
        if not names:
            st.warning("No datasets registered.")
            return

        default_ds_idx = names.index("identity") if "identity" in names else 0
        selected = st.selectbox("Dataset", options=names, key="dg_dataset_name", index=default_ds_idx)
        cls = DATASET_FUNCTIONS[selected]
        d_fixed = 2

        col_definition, col_options = st.columns([1, 1], gap="large")

        # Execute options column first so sliders exist before cache / configure_data.
        with col_options:
            with st.container(border=True):
                st.markdown("**General Dataset Options**")
                num_samples = st.slider(
                    "Samples",
                    min_value=5,
                    max_value=50,
                    value=10,
                    step=1,
                    key="dg_num_samples",
                )
                sample_mode = st.radio(
                    "Input sampling",
                    options=(InputSamplingMode.RNG, InputSamplingMode.UNIFORM),
                    format_func=lambda m: (
                        "RNG (dataset default)" if m == InputSamplingMode.RNG else "UNIFORM on [0, 1)"
                    ),
                    horizontal=True,
                    key="dg_sample_mode",
                )

                add_noise = st.checkbox("Add Gaussian Noise", value=False, key="dg_add_gaussian_noise")

                noise_mode = None
                noise_mu = 0.0
                noise_sigma = 0.1
                noise_seed = 0
                if add_noise:
                    add_noise_to_inputs = st.checkbox("Add Noise to Inputs", value=True, key="dg_add_noise_to_inputs")
                    add_noise_to_targets = st.checkbox(
                        "Add Noise to Targets", value=True, key="dg_add_noise_to_targets"
                    )

                    if add_noise_to_inputs and add_noise_to_targets:
                        noise_mode = "both"
                    elif add_noise_to_inputs:
                        noise_mode = "inputs"
                    elif add_noise_to_targets:
                        noise_mode = "targets"

                    with st.container(border=True):
                        col1, col2, col3 = st.columns([1, 1, 1], gap="small")
                        with col1:
                            noise_mu = st.slider(
                                "Noise Mean",
                                min_value=0.0,
                                max_value=1.0,
                                value=0.0,
                                step=0.01,
                                key="dg_noise_mu",
                            )
                        with col2:
                            noise_sigma = st.slider(
                                "Noise Standard Deviation",
                                min_value=0.0,
                                max_value=1.0,
                                value=0.1,
                                step=0.01,
                                key="dg_noise_sigma",
                            )
                        with col3:
                            noise_seed = st.number_input(
                                "Noise Seed",
                                value=0,
                                min_value=0,
                                max_value=1000000,
                                step=1,
                                key="dg_noise_seed",
                            )

                st.divider()
                st.markdown("**Dataset Specific Options**")
                st.markdown("Coming soon...")

        with col_definition:
            with st.container(border=True):
                render_glossary_node(cls.get_glossary_node(), "", as_expander=False)
            if st.button("Regenerate Inputs", type="primary", key="dg_regenerate_inputs"):
                st.session_state.pop("dg_clean_X", None)

        cache_key = (selected, d_fixed, int(num_samples), sample_mode.value)
        if st.session_state.get("dg_input_cache_key") != cache_key:
            st.session_state.pop("dg_clean_X", None)
        st.session_state["dg_input_cache_key"] = cache_key

        try:
            ds = build_registered_dataset(
                selected,
                d=d_fixed,
                num_samples=int(num_samples),
                scale=1.0,
                noise_mode=noise_mode,
                noise_mu=noise_mu,
                noise_sigma=noise_sigma,
                noise_seed=noise_seed,
            )
            clean = ds.configure_data(
                st.session_state.get("dg_clean_X"),
                sample_mode=sample_mode,
            )
            st.session_state["dg_clean_X"] = clean
        except (ValueError, TypeError, RuntimeError) as err:
            st.error(f"Could not build dataset: {err}")
            return

        data = ds.get_data()
        x_arr = np.asarray(data["X"])
        y_arr = np.asarray(data["Y"])
        x_pairs, x_xl, x_yl = _scatter_pairs_from_2d(x_arr)
        y_pairs, y_xl, y_yl = _scatter_pairs_from_2d(y_arr)
        opt_in = _echarts_scatter_option(
            title="Input",
            series_data=x_pairs,
            x_label=x_xl,
            y_label=x_yl,
        )
        opt_out = _echarts_scatter_option(
            title="Outputs",
            series_data=y_pairs,
            x_label=y_xl,
            y_label=y_yl,
        )

        col_chart_input, col_chart_output = st.columns([1, 1], gap="large")
        with col_chart_input:
            st_echarts(options=opt_in, height="360px")
        with col_chart_output:
            st_echarts(options=opt_out, height="360px")
