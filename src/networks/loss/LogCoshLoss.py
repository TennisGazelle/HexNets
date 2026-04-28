import numpy as np

from networks.loss.loss import BaseLoss
from hexnets_web.glossary_types import GlossaryNode


class LogCoshLoss(BaseLoss, display_name="log_cosh"):

    @classmethod
    def get_glossary_node(cls) -> GlossaryNode:
        return GlossaryNode(
            title="Log-cosh loss",
            aliases=("log_cosh", "LogCoshLoss", "log cosh"),
            english=(
                "Smooth approximation to L1: mean(log(cosh(y_pred − y_true))) over elements. "
                "Behaves like **½ a²** for small errors and like **|a|** for large ones. "
                "**calc_delta** returns **tanh(y_pred − y_true)** elementwise (see code for alignment with your trainer)."
            ),
            math_latex=r"\ell = \frac{1}{nk}\sum_{i,j} \log\cosh(y_{\text{pred},ij}-y_{\text{true},ij})",
            tags=("regression-compatible",),
        )

    def calc_loss(self, y_true, y_pred):
        diff = y_pred - y_true
        log_cosh = np.log(np.cosh(diff))
        return np.mean(log_cosh)

    def calc_delta(self, y_true, y_pred):
        diff = y_pred - y_true
        return np.tanh(diff)
