import numpy as np
import matplotlib.pyplot as plt
import pickle
from abc import ABC, abstractmethod
import pathlib

# =============================
# Activations (pluggable)
# =============================
class BaseActivations:
    def __init__(self, name="SIGMOID"):
        self._name = name.upper()
        self.set_activation()

    def set_activation(self, name=None):
        if name is not None:
            self._name = name.upper()
        if self._name not in {"SIGMOID", "RELU", "IDENTITY"}:
            raise ValueError(f"Unsupported activation: {self._name}")
        if self._name == "SIGMOID":
            self.activate = self._sigmoid
            self.deactivate = self._sigmoid_deriv_from_output
        elif self._name == "RELU":
            self.activate = self._relu
            self.deactivate = self._relu_deriv_from_output
        else:  # IDENTITY
            self.activate = self._identity
            self.deactivate = self._identity_deriv_from_output

    def name(self):
        return self._name

    # ---- funcs ----
    @staticmethod
    def _sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    @staticmethod
    def _sigmoid_deriv_from_output(y):
        return y * (1.0 - y)

    @staticmethod
    def _relu(x):
        return np.maximum(0.0, x)

    @staticmethod
    def _relu_deriv_from_output(y):
        return (y > 0.0).astype(float)

    @staticmethod
    def _identity(x):
        return x

    @staticmethod
    def _identity_deriv_from_output(y):
        return np.ones_like(y)


# =============================
# Base class (shared API)
# =============================
class BaseNeuralNetwork(ABC):
    def __init__(self):
        self._activations = BaseActivations("SIGMOID")

    # ---- activation API ----
    def setActivationFunc(self, name):
        self._activations.set_activation(name)

    def getActivationFunc(self):
        return self._activations.name()

    # ---- required NN API ----
    @abstractmethod
    def save(self, filepath):
        pass

    @abstractmethod
    def load(self, filepath):
        pass

    @abstractmethod
    def train(self, data, **kwargs):
        pass

    @abstractmethod
    def test(self, x, **kwargs):
        pass

    def graph(self, activation_only=True, save_to_file_path=None):
        self._graphW(activation_only=activation_only, save_to_file_path=save_to_file_path)


# =============================
# Hexagonal Neural Network
# =============================
class HexagonalNeuralNetwork(BaseNeuralNetwork):
    def __init__(self, n, random_init=True, lr=0.0005, mode="regression"):
        super().__init__()
        self.n = n
        self.layer_indices = self._get_default_layer_indices(n)
        self.total_nodes = sum(len(l) for l in self.layer_indices)
        self.W = self._init_weight_matrix(random_init=random_init)
        self.learning_rate = lr
        self.mode = mode
        self._activations.set_activation("SIGMOID")

    # ---------- structure helpers ----------
    def _set_activation_func(self, name):
        self._activations.set_activation(name)

    def _hex_layer_sizes(self, n):
        return list(range(n, 2*n)) + list(range(2*n - 2, n - 1, -1))

    def _get_default_layer_indices(self, n):
        sizes = self._hex_layer_sizes(n)
        indices = []
        start = 0
        for size in sizes:
            indices.append(list(range(start, start + size)))
            start += size
        return indices

    # ---------- weights (dense mask on adj blocks, zeros elsewhere) ----------
    def _init_weight_matrix(self, random_init=True):
        W = np.zeros((self.total_nodes, self.total_nodes))
        if not random_init:
            return W
        rng = np.random.default_rng()
        for i in range(len(self.layer_indices) - 1):
            src = self.layer_indices[i]
            dst = self.layer_indices[i + 1]
            fan_in = len(src)
            fan_out = len(dst)
            std = np.sqrt(2.0 / (fan_in + fan_out))
            for u in src:
                for v in dst:
                    W[u, v] = rng.normal(0.0, std)
        return W

    # ---------- forward / backward ----------
    def _forward_internal(self, x):
        """
        mode: "regression" | "classification"
        Hidden: use current activation
        Output: identity if regression, sigmoid if classification
        Returns (activations_per_step)
        """
        activations = [x.copy()]
        a = x.copy()
        steps = len(self.layer_indices) - 1
        for k in range(steps):
            z = self.W @ a
            if k < steps - 1:
                a = self._activations.activate(z)
            else:
                if self.mode == "classification":
                    a = self._activations.activate(z)
                else:
                    a = self._activations.activate(z)
            activations.append(a.copy())
        return activations

    def _backward_internal(self, activations, target_full):
        grads_W = np.zeros_like(self.W)
        y_hat = activations[-1]

        if self.mode == "classification":
            # BCE with sigmoid output: delta = y_hat - y
            delta = (self._activations.activate(y_hat) - target_full)
        else:
            # MSE with identity output: delta = (y_hat - y) * 1
            delta = (y_hat - target_full)

        for i in reversed(range(len(self.layer_indices) - 1)):
            src = self.layer_indices[i]
            dst = self.layer_indices[i + 1]
            a_prev = activations[i]

            for u in src:
                au = a_prev[u]
                if au != 0.0:
                    for v in dst:
                        grads_W[u, v] += delta[v] * au

            if i > 0:
                # propagate to previous layer
                new_delta = np.zeros(self.total_nodes)
                for u in src:
                    s = 0.0
                    for v in dst:
                        s += self.W[u, v] * delta[v]
                    # derivative of hidden activation using post-activation a_prev[u]
                    gprime = self._activations.deactivate(a_prev[u])
                    new_delta[u] = s * gprime
                delta = new_delta

        self.W -= self.learning_rate * grads_W

    # ---------- public API ----------
    def forward(self, x):
        return self._forward_internal(x)

    def backward(self, activations, target_full):
        self._backward_internal(activations, target_full)

    def _training_step(self, x_in, y_out):
        x0 = np.zeros(self.total_nodes)
        x0[self.layer_indices[0]] = x_in
        y_full = np.zeros(self.total_nodes)
        y_full[self.layer_indices[-1]] = y_out
        acts = self._forward_internal(x0)
        self._backward_internal(acts, y_full)
        return acts

    # ---- metrics ----
    @staticmethod
    def _mse(y_pred, y_true):
        return np.mean(np.square(y_pred - y_true))

    @staticmethod
    def _bce(y_pred_sigmoid, y_true, eps=1e-7):
        p = np.clip(y_pred_sigmoid, eps, 1 - eps)
        return -np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p))

    def _compute_epoch_metrics(self, y_pred, y_true, threshold=0.5):
        if self.mode == "classification":
            p = 1.0 / (1.0 + np.exp(-y_pred))
            loss = self._bce(p, y_true)
            acc = np.mean((p >= threshold) == (y_true >= 0.5))
            return loss, acc
        # regression
        loss = self._mse(y_pred, y_true)
        # “accuracy” proxy: inverse normalized error
        mae = np.mean(np.abs(y_pred - y_true))
        acc = 1.0 / (1.0 + mae)
        return loss, acc

    # ---- training (non-animated, optional graphs) ----
    def train(self, data, epochs=1, threshold=0.5, product_graphs=True):
        losses = []
        accs = []
        for _ in range(epochs):
            total_loss = 0.0
            total_acc = 0.0
            count = 0
            for x_in, y_out in data:
                acts = self._training_step(x_in, y_out)
                y_pred = acts[-1][self.layer_indices[-1]]
                loss, acc = self._compute_epoch_metrics(y_pred, y_out, threshold)
                total_loss += loss
                total_acc += acc
                count += 1
            print(f"Epoch {_ + 1} loss: {total_loss}, acc: {total_acc / len(data)}")
            losses.append(total_loss)
            accs.append(total_acc / len(data))

        if product_graphs:
            self._plot_training_curves(losses, accs, title_suffix=f"mode={self.mode}")
        return np.array(losses), np.array(accs)

    # ---- animated training (kept; uses same metrics) ----
    def train_animated(self, data, epochs=25, threshold=0.5, pause=0.05):
        losses = []
        accs = []
        fig_loss = plt.figure(figsize=(6, 4))
        ax_loss = fig_loss.add_subplot(111)
        line_loss, = ax_loss.plot([], [])
        ax_loss.set_title("Training Loss")
        ax_loss.set_xlabel("Epoch")
        ax_loss.set_ylabel("Loss")
        ax_loss.grid(True)

        fig_acc = plt.figure(figsize=(6, 4))
        ax_acc = fig_acc.add_subplot(111)
        line_acc, = ax_acc.plot([], [])
        ax_acc.set_title("Training Accuracy")
        ax_acc.set_xlabel("Epoch")
        ax_acc.set_ylabel("Accuracy")
        ax_acc.set_ylim(0, 1)
        ax_acc.grid(True)

        for epoch in range(epochs):
            total_loss = 0.0
            total_acc = 0.0
            count = 0
            for x_in, y_out in data:
                acts = self._training_step(x_in, y_out)
                y_pred = acts[-1][self.layer_indices[-1]]
                loss, acc = self._compute_epoch_metrics(y_pred, y_out, self.mode, threshold)
                total_loss += loss
                total_acc += acc
                count += 1

            losses.append(total_loss / max(1, count))
            accs.append(total_acc / max(1, count))

            line_loss.set_data(np.arange(1, len(losses)+1), losses)
            ax_loss.relim()
            ax_loss.autoscale_view()
            fig_loss.canvas.draw()
            plt.pause(pause)

            line_acc.set_data(np.arange(1, len(accs)+1), accs)
            ax_acc.relim()
            ax_acc.autoscale_view()
            fig_acc.canvas.draw()
            plt.pause(pause)

        return np.array(losses), np.array(accs)

    # ---- plotting helpers ----
    def _plot_training_curves(self, losses, accs, title_suffix=""):
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(np.arange(1, len(losses)+1), losses)
        ax.set_title(f"Loss {title_suffix}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(True)
        plt.show()

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(np.arange(1, len(accs)+1), accs)
        ax.set_title(f"Accuracy {title_suffix}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0, 1)
        ax.grid(True)
        plt.show()

    # ---- test ----
    def test(self, x_input, threshold=0.5):
        x0 = np.zeros(self.total_nodes)
        x0[self.layer_indices[0]] = x_input
        acts = self._forward_internal(x0)
        y = acts[-1][self.layer_indices[-1]]
        if self.mode == "classification":
            p = 1.0 / (1.0 + np.exp(-y))
            return (p >= threshold).astype(int), p
        return y, y

    # ---- persistence ----
    def save(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump({'n': self.n, 'W': self.W, 'act': self.getActivationFunc()}, f)

    def load(self, filepath):
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
            self.n = state['n']
            self.layer_indices = self._get_default_layer_indices(self.n)
            self.total_nodes = sum(len(l) for l in self.layer_indices)
            self.W = state['W']
            self.setActivationFunc(state.get('act', 'SIGMOID'))

    # ---- graphs of structure / weights ----
    def _graphW(self, activation_only=True, save_to_file_path=None):
        matrix = (self.W != 0).astype(int) if activation_only else self.W
        plt.figure(figsize=(7, 7))
        plt.imshow(matrix, interpolation='none')
        plt.title(("Activation Structure" if activation_only else "Weight Matrix") + f" (n={self.n})")
        plt.xticks(np.arange(self.total_nodes))
        plt.yticks(np.arange(self.total_nodes))
        plt.grid(visible=True)
        plt.colorbar()
        if save_to_file_path is not None:
            plt.savefig(save_to_file_path)
        else:
            plt.show()

    # ---- static reference graphs ----
    @staticmethod
    def show_reference_graphs(save_to_file_path=None):
        def hex_layer_sizes(n):
            return list(range(n, 2*n)) + list(range(2*n - 2, n - 1, -1))
        def get_default_layer_indices(n):
            sizes = hex_layer_sizes(n)
            idx = []
            s = 0
            for size in sizes:
                idx.append(list(range(s, s + size)))
                s += size
            return idx
        def build_adj(layer_indices):
            N = sum(len(l) for l in layer_indices)
            A = np.zeros((N, N), dtype=int)
            for i in range(len(layer_indices) - 1):
                for u in layer_indices[i]:
                    for v in layer_indices[i+1]:
                        A[u, v] = 1
            return A
        A2 = build_adj(get_default_layer_indices(2))
        A3 = build_adj(get_default_layer_indices(3))
        A4 = build_adj(get_default_layer_indices(4))
        fig, axs = plt.subplots(2, 2, figsize=(12, 12))
        axs[0, 0].imshow(A2, cmap='Blues', interpolation='none')
        axs[0, 0].set_title("n = 2")
        axs[0, 0].grid(visible=True, color='black', linewidth=0.5)
        axs[0, 1].imshow(A3, cmap='Oranges', interpolation='none')
        axs[0, 1].set_title("n = 3")
        axs[0, 1].grid(visible=True, color='black', linewidth=0.5)
        axs[1, 0].imshow(A4, cmap='Greens', interpolation='none')
        axs[1, 0].set_title("n = 4")
        axs[1, 0].grid(visible=True, color='black', linewidth=0.5)
        axs[1, 1].axis('off')
        axs[1, 1].text(0.5, 0.5, "Hexagonal Adjacency\nTop Left: n=2, Top Right: n=3\nBottom Left: n=4", ha='center', va='center', fontsize=14, wrap=True)
        plt.suptitle("Reference Hexagonal Adjacency Matrices", fontsize=16)
        plt.tight_layout()
        if save_to_file_path is not None:
            plt.savefig(save_to_file_path)
        else:
            plt.show()

def main():

    # ---------- Demo with synthetic data ----------
    # We'll create a tiny binary task on n=2 (output matches input for half the samples)
    # HexagonalNeuralNetwork.show_reference_graphs(save_to_file_path=pathlib.Path("hexnet_reference_graphs.png"))
    n = 3
    regression_net = HexagonalNeuralNetwork(n=n, random_init=True, lr=0.001, mode="regression")
    regression_net.graph(activation_only=False, save_to_file_path=pathlib.Path(f"hexnet_graph_w_n{n}_uninitialized.png"))

    # classification_net = HexagonalNeuralNetwork(n=n, random_init=True, lr=0.001, mode="classification")
    # classification_net.graph(activation_only=False, save_to_file_path=pathlib.Path(f"hexnet_graph_w_n{n}_uninitialized.png"))

    # -- Training Data --
    # (Procedural, regression sets)

    # simple dataset: inputs in {0,1}^n and targets = inputs (identity) for demo
    # train_samples = 1000
    # X = (np.random.rand(train_samples, n) * 2 - 1).astype(float)
    # # Y = X.copy()
    # Y = X.copy() * 2
    # data = list(zip(X, Y))

    # print(data)

    # # a simple dataset: Y = 2X
    # data = np.array([
    #     # ([0.1, 0.2], [0.2, 0.4]),
    #     # ([0.5, 0.6], [1.0, 1.2]),
    #     # ([0.3, 0.4], [0.6, 0.8]),
    #     # ([0.7, 0.8], [1.4, 1.6]),
    #     # ([0.9, 1.0], [1.8, 2.0])
    #     ([x0, x1], [2 * x0, 2 * x1]) 
    #     for x0 in np.arange(-1.0, 1.0, 0.1) for x1 in np.arange(-1.0, 1.0, 0.1)
    # ])

    # print(data)

    # -- Training --
    # (Procedural, classification sets)
    # Dataset: multi-label identity with some noise to avoid symmetry traps
    train_samples = 1000
    X = (np.random.rand(train_samples, n) > 0.5).astype(bool)
    noise_mask = (np.random.rand(train_samples, n) < 0.1)
    Y = (X ^ noise_mask).astype(float)
    data = list(zip(X, Y))

    print("Training...")
    losses, accs = regression_net.train(data, epochs=250, threshold=1)

    # Show final structure & weights for sanity
    # net.graph(activation_only=True)
    regression_net.graph(activation_only=False, save_to_file_path=pathlib.Path(f"hexnet_graph_w_n{n}_trained.png"))


if __name__ == "__main__":
    main()

