# Re-run after state reset
import numpy as np
import matplotlib.pyplot as plt
import pickle
from abc import ABC, abstractmethod

class BaseNeuralNetwork(ABC):
    @abstractmethod
    def save(self, filepath): pass
    @abstractmethod
    def load(self, filepath): pass
    @abstractmethod
    def train(self, data): pass
    @abstractmethod
    def test(self, x): pass
    def graph(self, activation_only=True): self._graphW(activation_only=activation_only)

class HexagonalNeuralNetwork(BaseNeuralNetwork):
    def __init__(self, n, random_init=True, lr=0.05):
        self.n = n
        self.layer_indices = self._get_default_layer_indices(n)
        self.total_nodes = sum(len(l) for l in self.layer_indices)
        self.W = self._init_weight_matrix(random_init=random_init)
        self.learning_rate = lr

    def _hex_layer_sizes(self, n):
        return list(range(n, 2*n)) + list(range(2*n - 2, n - 1, -1))

    def _get_default_layer_indices(self, n):
        sizes = self._hex_layer_sizes(n)
        indices = []; start = 0
        for size in sizes:
            indices.append(list(range(start, start + size)))
            start += size
        return indices

    def _init_weight_matrix(self, random_init=True):
        W = np.zeros((self.total_nodes, self.total_nodes))
        rng = np.random.default_rng()
        if not random_init: return W
        for i in range(len(self.layer_indices) - 1):
            src = self.layer_indices[i]; dst = self.layer_indices[i + 1]
            fan_in = len(src); fan_out = len(dst)
            std = np.sqrt(2.0 / (fan_in + fan_out))
            for u in src:
                for v in dst:
                    W[u, v] = rng.normal(0.0, std)
        return W

    @staticmethod
    def _sigmoid(x): return 1.0 / (1.0 + np.exp(-x))
    @staticmethod
    def _sigmoid_deriv_from_output(sig_y): return sig_y * (1.0 - sig_y)

    def forward(self, x):
        activations = [x.copy()]; a = x.copy()
        for _ in range(len(self.layer_indices) - 1):
            a = self._sigmoid(self.W @ a)
            activations.append(a.copy())
        return activations

    def backward(self, activations, target_full):
        grads_W = np.zeros_like(self.W)
        y_hat = activations[-1]
        delta = (y_hat - target_full)  # sigmoid + BCE

        for i in reversed(range(len(self.layer_indices) - 1)):
            src = self.layer_indices[i]; dst = self.layer_indices[i + 1]
            a_prev = activations[i]
            # dW
            for u in src:
                au = a_prev[u]
                if au == 0:  # small speedup; not necessary
                    continue
                for v in dst:
                    grads_W[u, v] += delta[v] * au
            # backprop delta
            if i > 0:
                new_delta = np.zeros(self.total_nodes)
                for u in src:
                    s = 0.0
                    for v in dst:
                        s += self.W[u, v] * delta[v]
                    new_delta[u] = s * self._sigmoid_deriv_from_output(a_prev[u])
                delta = new_delta

        self.W -= self.learning_rate * grads_W

    def train(self, data, epochs=1, multi_label=True, threshold=0.5):
        for _ in range(epochs):
            for x_in, y_out in data:
                x0 = np.zeros(self.total_nodes); x0[self.layer_indices[0]] = x_in
                y_full = np.zeros(self.total_nodes); y_full[self.layer_indices[-1]] = y_out
                acts = self.forward(x0); self.backward(acts, y_full)

    def test(self, x_input, multi_label=True, threshold=0.5):
        x0 = np.zeros(self.total_nodes); x0[self.layer_indices[0]] = x_input
        acts = self.forward(x0); y = acts[-1][self.layer_indices[-1]]
        if multi_label: return (y >= threshold).astype(int), y
        else: return np.argmax(y), y

    def save(self, filepath):
        with open(filepath, 'wb') as f: pickle.dump({'n': self.n, 'W': self.W}, f)
    def load(self, filepath):
        with open(filepath, 'rb') as f:
            state = pickle.load(f); self.n = state['n']
            self.layer_indices = self._get_default_layer_indices(self.n)
            self.total_nodes = sum(len(l) for l in self.layer_indices); self.W = state['W']

    def _graphW(self, activation_only=True):
        matrix = (self.W != 0).astype(int) if activation_only else self.W
        plt.figure(figsize=(7, 7)); plt.imshow(matrix, interpolation='none')
        plt.title(("Activation Structure" if activation_only else "Weight Matrix") + f" (n={self.n})")
        plt.xticks(np.arange(self.total_nodes)); plt.yticks(np.arange(self.total_nodes))
        plt.grid(visible=True); plt.colorbar(); plt.show()

    def train_animated(self, data, epochs=25, multi_label=True, threshold=0.5, pause=0.05):
        losses, accs = [], []
        fig_loss = plt.figure(figsize=(6, 4)); ax_loss = fig_loss.add_subplot(111)
        line_loss, = ax_loss.plot([], []); ax_loss.set_title("Training Loss (BCE)")
        ax_loss.set_xlabel("Epoch"); ax_loss.set_ylabel("Loss"); ax_loss.grid(True)

        fig_acc = plt.figure(figsize=(6, 4)); ax_acc = fig_acc.add_subplot(111)
        line_acc, = ax_acc.plot([], []); ax_acc.set_title("Training Accuracy")
        ax_acc.set_xlabel("Epoch"); ax_acc.set_ylabel("Accuracy"); ax_acc.set_ylim(0, 1); ax_acc.grid(True)

        for epoch in range(epochs):
            total_loss = 0.0; total_acc = 0.0; count = 0
            for x_in, y_out in data:
                x0 = np.zeros(self.total_nodes); x0[self.layer_indices[0]] = x_in
                y_full = np.zeros(self.total_nodes); y_full[self.layer_indices[-1]] = y_out
                acts = self.forward(x0); y_pred = acts[-1][self.layer_indices[-1]]

                eps = 1e-7; y_clip = np.clip(y_pred, eps, 1 - eps)
                bce = -np.mean(y_out * np.log(y_clip) + (1 - y_out) * np.log(1 - y_clip))
                total_loss += bce

                if multi_label:
                    pred_bin = (y_pred >= threshold).astype(int)
                    acc = np.mean(pred_bin == (y_out >= 0.5))
                else:
                    acc = float(np.argmax(y_pred) == np.argmax(y_out))
                total_acc += acc

                self.backward(acts, y_full); count += 1

            losses.append(total_loss / max(1, count)); accs.append(total_acc / max(1, count))
            line_loss.set_data(np.arange(1, len(losses)+1), losses); ax_loss.relim(); ax_loss.autoscale_view()
            fig_loss.canvas.draw(); plt.pause(pause)
            line_acc.set_data(np.arange(1, len(accs)+1), accs); ax_acc.relim(); ax_acc.autoscale_view()
            fig_acc.canvas.draw(); plt.pause(pause)

        return np.array(losses), np.array(accs)

# Demo
np.random.seed(3)
n = 3
model = HexagonalNeuralNetwork(n=n, random_init=True, lr=0.2)

# Dataset: multi-label identity with some noise to avoid symmetry traps
X = (np.random.rand(256, n) > 0.5).astype(float)
# add label noise off-by-one 10% to make it non-trivial
noise = (np.random.rand(256, n) < 0.1).astype(float)
Y = (X ^ noise.astype(bool)).astype(float)

dataset = list(zip(X, Y))

losses, accs = model.train_animated(dataset, epochs=25, threshold=0.5, pause=0.03)

# Final quick sanity test
pred_bin, logits = model.test(np.array([1., 0., 1.]), threshold=0.5)
pred_bin, logits
