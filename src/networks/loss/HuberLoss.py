import numpy as np

from networks.loss.loss import BaseLoss
from hexnets_web.glossary_types import GlossaryNode


class HuberLoss(BaseLoss, display_name="huber"):
    @classmethod
    def get_glossary_node(cls) -> GlossaryNode:
        return GlossaryNode(
            title="Huber loss",
            aliases=("huber", "HuberLoss", "smooth l1", "pseudo-huber"),
            english=(
                "Robust regression loss: quadratic when |y_pred − y_true| ≤ **delta_threshold**, "
                "else linear beyond that threshold. Constructor default **delta_threshold** is 1.0. "
                "The reported loss is the **mean** over elements (same scalar style as MSE here)."
            ),
            math_latex=r"\ell_\delta(a)=\begin{cases}\frac12 a^2 & |a|\le\delta \\ \delta(|a|-\frac12\delta) & |a|>\delta\end{cases},\quad a=y_{\text{pred}}-y_{\text{true}}",
            example="With delta=1, error 0.5 contributes 0.125 to the quadratic branch; error 3 contributes 2.5 on the linear branch.",
            tags=("regression-compatible", "outlier-robust"),
        )

    def __init__(self, delta_threshold=1.0):
        self.delta_threshold = delta_threshold

    def calc_loss(self, y_true, y_pred):
        diff = y_pred - y_true
        abs_diff = np.abs(diff)
        quadratic = 0.5 * diff**2
        linear = self.delta_threshold * (abs_diff - 0.5 * self.delta_threshold)
        loss = np.where(abs_diff <= self.delta_threshold, quadratic, linear)
        # match MSE: return a scalar mean
        return np.mean(loss)

    def calc_delta(self, y_true, y_pred):
        diff = y_pred - y_true
        abs_diff = np.abs(diff)
        # elementwise grad (not normalized by N, to match your MSE.delta)
        return (
            np.where(
                abs_diff <= self.delta_threshold,
                diff,
                self.delta_threshold * np.sign(diff),
            )
            / y_true.shape[0]
        )
