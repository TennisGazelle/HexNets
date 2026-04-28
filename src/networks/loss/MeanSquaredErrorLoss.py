import numpy as np

from networks.loss.loss import BaseLoss
from hexnets_web.glossary_types import GlossaryNode


class MeanSquaredErrorLoss(BaseLoss, display_name="mean_squared_error"):

    @classmethod
    def get_glossary_node(cls) -> GlossaryNode:
        return GlossaryNode(
            title="Mean squared error (MSE)",
            aliases=(
                "mean_squared_error",
                "mse",
                "MSE",
                "MeanSquaredErrorLoss",
            ),
            english=(
                "Per-example squared error averaged over all elements: mean((y_pred − y_true)²). "
                "**calc_delta** returns (2/N)·(y_pred − y_true) with **N = y_true.shape[0]** (batch size), "
                "matching the gradient shape used elsewhere in this trainer."
            ),
            math_latex=r"\ell = \frac{1}{nk}\sum_{i,j} (y_{\text{pred},ij}-y_{\text{true},ij})^2",
            example="If predictions match targets everywhere, loss is 0.",
            tags=("regression-compatible",),
        )

    def calc_loss(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    def calc_delta(self, y_true, y_pred):
        return (2 / y_true.shape[0]) * (y_pred - y_true)
