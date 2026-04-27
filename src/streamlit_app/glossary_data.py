"""
Glossary tree data (no Streamlit imports).
"""

from __future__ import annotations

from dataclasses import dataclass, field

@dataclass
class GlossaryNode:
    title: str
    english: str
    aliases: tuple[str, ...] = ()
    math_latex: str | None = None
    example: str | None = None
    children: tuple["GlossaryNode", ...] = ()
    search_blob: str = field(default="", repr=False)


def _fill_search_blob(node: GlossaryNode) -> None:
    parts = [node.title, node.english, *node.aliases]
    if node.math_latex:
        parts.append(node.math_latex)
    if node.example:
        parts.append(node.example)
    for c in node.children:
        _fill_search_blob(c)
        parts.append(c.search_blob)
    node.search_blob = " ".join(parts).lower()


def _build_glossary_root() -> list[GlossaryNode]:
    identity = GlossaryNode(
        title="Identity dataset",
        aliases=("identity", "type=identity"),
        english=(
            "Training pairs where each target **y** equals the corresponding input **x**. "
            "In code this is `IdentityDataset`: it uses the same random **x** in [-1, 1]^d as "
            "`LinearScaleDataset` but with scale 1, so **y = x**."
        ),
        math_latex=r"y = x \quad \text{(elementwise)}",
        example="For d=3, one sample might be x = [0.2, -0.5, 0.1] and y = [0.2, -0.5, 0.1].",
        children=(),
    )
    linear = GlossaryNode(
        title="Linear (scaled) dataset",
        aliases=("linear", "type=linear", "linear_scale", "LinearScaleDataset"),
        english=(
            "Inputs **x** are drawn uniformly in [-1, 1]^d; targets are **y = scale · x** with a "
            "scalar `scale` (default 2.0 in some CLI paths). The Streamlit **Train Network** "
            "button uses the identity variant (`get_dataset(..., type='identity')`), not this one."
        ),
        math_latex=r"y = s \cdot x",
        example="With scale=2, if x = [0.5, -1.0] then y = [1.0, -2.0].",
        children=(),
    )
    datasets = GlossaryNode(
        title="Datasets",
        aliases=("data", "training data", "samples"),
        english=(
            "A dataset here is an iterable of (input, target) pairs used for training. "
            "Each vector has length **n** (the network’s node count). Expand the entries below "
            "for the kinds used in this project."
        ),
        children=(identity, linear),
    )

    loss = GlossaryNode(
        title="Loss (epoch)",
        aliases=("mse", "huber", "mean_squared_error"),
        english=(
            "The number minimized by gradient descent: the **mean** over the training set of "
            "whatever loss class is configured (e.g. MSE between **y_pred** and **y_target** on "
            "the output nodes). This is **not** the same as regression score."
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
            "see the **Training metrics** expander on the **Network Explorer** tab."
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
        english="Nonlinearity applied in hidden layers; chosen from the **Activation** dropdown in Network Explorer.",
        children=(),
    )
    param_loss = GlossaryNode(
        title="Loss function (config)",
        aliases=("loss", "objective"),
        english="The **Loss** dropdown picks which **Loss** class is used when computing epoch loss (e.g. MSE).",
        children=(),
    )
    params = GlossaryNode(
        title="Hex network parameters",
        aliases=("hyperparameters", "controls"),
        english="Main knobs in the Network Explorer tab that define the `HexagonalNeuralNetwork` instance.",
        children=(param_n, param_r, param_act, param_loss),
    )

    root = [
        datasets,
        metrics_parent,
        params,
    ]
    for n in root:
        _fill_search_blob(n)
    return root


GLOSSARY_ROOT: list[GlossaryNode] = _build_glossary_root()
