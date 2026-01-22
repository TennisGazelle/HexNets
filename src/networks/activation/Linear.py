from networks.activation.activations import BaseActivation


class Linear(BaseActivation, display_name="linear"):
    def activate(self, x):
        return x

    def deactivate(self, x):
        return 1.0