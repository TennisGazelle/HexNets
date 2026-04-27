import streamlit as st


def render_training_metrics_expander() -> None:
    """Collapsed expander with formulas, caveats, toy and session last-run metrics."""
    with st.expander("Training metrics — formulas, caveats, examples", expanded=False):
        st.markdown(
            "Notation: **y_target** and **y_pred** are length-**n** vectors per example; "
            "**count** is the number of examples in the epoch."
        )

        st.subheader("Loss")
        st.markdown(
            "The **epoch loss** is the mean over training samples of the configured loss "
            "(e.g. MSE) between predictions and targets on the output nodes. "
            "It is the objective minimized by backprop; it is **not** the same as regression score."
        )

        st.subheader("Regression score")
        st.markdown(
            "Per example: RMSE over output dimensions, then **score = exp(-RMSE)**. "
            "Epoch value: **mean** of per-example scores. Range **(0, 1]**; **1** means zero error. "
            "This is **not** classification accuracy."
        )
        st.latex(
            r"\mathrm{RMSE}_i = \sqrt{\frac{1}{n}\sum_{j=1}^{n}(y_{\mathrm{pred},ij}-y_{\mathrm{true},ij})^2},\quad "
            r"s_i = e^{-\mathrm{RMSE}_i},\quad \text{regression\_score} = \frac{1}{\mathrm{count}}\sum_i s_i"
        )

        st.subheader("R²")
        st.markdown(
            "Aggregated over the epoch from running sums (see `Metrics.calc_regression_score_and_r2` in code). "
            "A small constant **1e-12** is added to the denominator for numerical stability."
        )
        st.latex(r"R^2 = 1 - \frac{\mathrm{SS}_{\mathrm{res}}}{\mathrm{SS}_{\mathrm{tot}} + 10^{-12}}")

        st.subheader("Adjusted R²")
        st.markdown(
            "Corrects R² for model complexity with **p** parameters (here often **p = n**). "
            "Let **N = count · n** (number of scalar target values). If **N ≤ p + 1**, adjusted R² is set equal to R²."
        )
        st.latex(r"\bar{R}^2 = 1 - (1-R^2)\frac{N-1}{N-p-1}")

        st.subheader("Caveats")
        st.markdown(
            "- **R²** and **adjusted R²** are **epoch-level** aggregates, not per-example curves.\n"
            "- Legacy artifacts using keys like **accuracy** are not interchangeable with these metric names."
        )

        st.subheader("Toy numeric example")
        st.markdown(
            "One example with **n = 2**: y_target = [0, 0], y_pred = [0, 1]. "
            "Squared errors [0, 1], mean 0.5, RMSE = √0.5 ≈ 0.707, **exp(-RMSE) ≈ 0.49**."
        )

        last = st.session_state.get("last_metrics")
        if last:
            st.subheader("Last run (this session)")
            st.write(
                {
                    "loss": last.get("loss"),
                    "regression_score": last.get("regression_score"),
                    "r_squared": last.get("r_squared"),
                    "adjusted_r_squared": last.get("adjusted_r_squared"),
                }
            )
        else:
            st.caption("Train the network once to see the last epoch’s metrics here.")
