"""
Glossary tree data (no Streamlit imports).
"""

from __future__ import annotations

from data.dataset import build_datasets_glossary_parent
from networks.activation.activations import build_activations_glossary_parent
from networks.learning_rate.learning_rate import build_learning_rates_glossary_parent
from networks.loss.loss import build_losses_glossary_parent
from hexnets_web.glossary_types import GlossaryNode


def _fill_search_blob(node: GlossaryNode) -> None:
    parts = [node.title, node.english, *node.aliases]
    if node.math_latex:
        parts.append(node.math_latex)
    if node.example:
        parts.append(node.example)
    if node.good_for:
        parts.append(node.good_for)
    if node.tags:
        parts.append(" ".join(node.tags))
    for c in node.children:
        _fill_search_blob(c)
        parts.append(c.search_blob)
    node.search_blob = " ".join(parts).lower()


def _build_glossary_root() -> list[GlossaryNode]:
    datasets = build_datasets_glossary_parent()
    loss_functions = build_losses_glossary_parent()
    learning_rates = build_learning_rates_glossary_parent()
    activations = build_activations_glossary_parent()

    loss = GlossaryNode(
        title="Loss (epoch)",
        aliases=("mse", "huber", "mean_squared_error"),
        english=(
            "The number minimized by gradient descent: the **mean** over the training set of "
            "whatever loss class is configured (e.g. MSE between **y_pred** and **y_target** on "
            "the output nodes). This is **not** the same as regression score. "
            "For definitions of each **Loss** class (MSE, Huber, …), see the top-level **Loss functions** glossary entry."
        ),
        math_latex=r"L_{\text{epoch}} = \frac{1}{N}\sum_{i=1}^{N} \ell(y^{(i)}_{\text{pred}}, y^{(i)}_{\text{true}})",
        example="If every sample has loss 0.04, the epoch loss is 0.04.",
        children=(),
    )
    reg_score = GlossaryNode(
        title="Regression score",
        aliases=("exp(-rmse)", "mean exp(-RMSE)", "regression_score"),
        english=(
            "For each example, RMSE is the root mean square error over output dimensions; "
            "the per-example score is exp(-RMSE). The logged **regression score** is the **mean** "
            "of those scores over examples in the epoch. It is bounded in (0, 1]; it is **not** "
            "classification accuracy."
        ),
        math_latex=r"\text{RMSE}_i = \sqrt{\frac{1}{n}\sum_j (y_{\text{pred},ij}-y_{\text{true},ij})^2},\quad s_i = e^{-\text{RMSE}_i}",
        example="If RMSE=0 for one sample, s_i=1. If RMSE=0.5, s_i≈0.61.",
        children=(),
    )
    r2 = GlossaryNode(
        title="R² (coefficient of determination)",
        aliases=("r_squared", "R-squared", "r2"),
        english=(
            "Computed once per **epoch** from accumulated sums over all scalar targets: "
            "residual sum of squares vs. total sum of squares about the mean. "
            "It is an aggregate over the epoch, not a per-example time series in the plots."
        ),
        math_latex=r"R^2 = 1 - \frac{\mathrm{SS}_{\mathrm{res}}}{\mathrm{SS}_{\mathrm{tot}} + 10^{-12}}",
        example="A value of 1 means perfect predictions; negative values mean the model is worse than predicting the mean target.",
        children=(),
    )
    adj_r2 = GlossaryNode(
        title="Adjusted R²",
        aliases=("adjusted_r_squared", "adjusted r2"),
        english=(
            "Penalizes R² when many parameters are fit relative to the number of scalar observations "
            "`count * n`. Here **p** is taken as **n** (output dims) in the trainer. "
            "If `count * n <= p + 1`, the code sets adjusted R² equal to R²."
        ),
        math_latex=r"\bar{R}^2 = 1 - (1-R^2)\frac{N-1}{N-p-1},\quad N = \text{count}\cdot n",
        example="When the model is complex and data is scarce, adjusted R² is often lower than R².",
        children=(),
    )
    metrics_parent = GlossaryNode(
        title="Training metrics",
        aliases=("metrics", "evaluation"),
        english=(
            "Quantities recorded each epoch during training. For formulas and caveats in one place, "
            "see the **Training metrics** expander on the **Network Explorer** page."
        ),
        children=(loss, reg_score, r2, adj_r2),
    )

    param_n = GlossaryNode(
        title="n (number of nodes)",
        aliases=("nodes", "num_dims", "dimensions"),
        english=(
            "Both input and output dimension for the hex network in this app: the slider "
            "**Number of nodes (n)**. Larger **n** means more nodes and a richer graph."
        ),
        children=(),
    )
    param_r = GlossaryNode(
        title="r (rotation)",
        aliases=("rotation index", "hex rotation"),
        english=(
            "One of six discrete rotations of the hex connectivity pattern (0–5). "
            "Changing **r** relabels the same abstract graph; see **Rotation Comparison** for visuals."
        ),
        children=(),
    )
    param_act = GlossaryNode(
        title="Activation function",
        aliases=("relu", "activation"),
        english=(
            "Chosen from the **Activation** dropdown in Network Explorer. "
            "See the top-level **Activations** glossary for each registered function (forward and backward behavior)."
        ),
        children=(),
    )
    param_loss = GlossaryNode(
        title="Loss function (config)",
        aliases=("loss", "objective"),
        english=(
            "The **Loss** dropdown selects which **BaseLoss** subclass computes epoch loss and deltas. "
            "See the top-level **Loss functions** glossary for MSE, Huber, and the other implementations."
        ),
        children=(),
    )
    params = GlossaryNode(
        title="Hex network parameters",
        aliases=("hyperparameters", "controls"),
        english="Main knobs in the Network Explorer page that define the `HexagonalNeuralNetwork` instance.",
        children=(param_n, param_r, param_act, param_loss),
    )

    root = [
        datasets,
        loss_functions,
        learning_rates,
        activations,
        metrics_parent,
        params,
    ]
    for n in root:
        _fill_search_blob(n)
    return root


GLOSSARY_ROOT: list[GlossaryNode] = _build_glossary_root()
