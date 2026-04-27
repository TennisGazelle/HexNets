import numpy as np

from networks.loss.loss import BaseLoss
from streamlit_app.glossary_types import GlossaryNode


class QuantileLoss(BaseLoss, display_name="quantile"):
    @classmethod
    def get_glossary_node(cls) -> GlossaryNode:
        return GlossaryNode(
            title="Quantile (pinball) loss",
            aliases=("quantile", "QuantileLoss", "pinball", "tau loss"),
            english=(
                "Asymmetric linear loss for quantile regression: mean(max(τ·d, (τ−1)·d)) with **d = y_pred − y_true** "
                "and **tau** in (0, 1) (default 0.5 gives the check loss centered at the median). "
                "**calc_delta** uses τ or τ−1 depending on the sign of **d**."
            ),
            math_latex=r"\rho_\tau(d)=d\cdot(\tau-\mathbf{1}_{d<0})",
            example="With tau=0.5, over-predicting and under-predicting by the same amount incur equal cost.",
            tags=("regression-compatible", "quantile"),
        )

    def __init__(self, tau=0.5):
        self.tau = tau

    def calc_loss(self, y_true, y_pred):
        diff = y_pred - y_true
        return np.mean(np.maximum(self.tau * diff, (self.tau - 1) * diff))

    def calc_delta(self, y_true, y_pred):
        diff = y_pred - y_true
        return np.where(diff >= 0, self.tau, self.tau - 1)
