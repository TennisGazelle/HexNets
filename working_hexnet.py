import numpy as np
import matplotlib.pyplot as plt
import pickle
from abc import ABC, abstractmethod

# === Base class (with graph) ===
class BaseNeuralNetwork(ABC):
    @abstractmethod
    def save(self, filepath):
        pass

    @abstractmethod
    def load(self, filepath):
        pass

    @abstractmethod
    def train(self, data):
        pass

    @abstractmethod
    def test(self, x):
        pass

    def graph(self, activation_only=True, detail=""):
        self._graphW(activation_only=activation_only, detail=detail)


# === Hexagonal Neural Network ===
class HexagonalNeuralNetwork(BaseNeuralNetwork):
    def __init__(self, n, random_init=True, lr=0.01):
        self.n = n
        self.layer_indices = self._get_default_layer_indices(n)
        self.total_nodes = sum(len(l) for l in self.layer_indices)
        self.W = self._init_weight_matrix(random_init=random_init)
        self.learning_rate = lr

    # --- structure helpers ---
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

    # --- weights ---
    def _init_weight_matrix(self, random_init=True):
        W = np.zeros((self.total_nodes, self.total_nodes))
        for i in range(len(self.layer_indices) - 1):
            for u in self.layer_indices[i]:
                for v in self.layer_indices[i + 1]:
                    W[u, v] = np.random.randn() * 2 if random_init else 0.0
        return W

    # --- activations ---
    def _relu(self, x, alpha=0.01):
        return np.maximum(0, x)
        # return np.where(x >= 0, x, alpha * x)

    def _relu_deriv(self, x, alpha=0.01):
        # x here is the post-activation value used in our forward history
        return (x > 0).astype(float)
        # return np.where(x > 0, 1.0, 0.0)

    # --- forward & backward ---
    def forward(self, x):
        activations = [x.copy()]
        a = x.copy()
        for i in range(len(self.layer_indices) - 1):
            z = self.W.T @ a
            if i < len(self.layer_indices) - 2:
                a = self._relu(z)
            else:
                a = z
            activations.append(a.copy())
        return activations

    def backward(self, activations, target):
        grads = np.zeros_like(self.W)
        diff = activations[-1] - target  # linear output layer
        delta_thr = 0.1
        delta = np.where(np.abs(diff) <= delta_thr, diff, delta_thr * np.sign(diff))
        # walk layers backward
        for i in reversed(range(len(self.layer_indices) - 1)):
            src_nodes = self.layer_indices[i]
            dst_nodes = self.layer_indices[i + 1]
            # weight grads: outer product between delta(dst) and activations(src)
            for u in src_nodes:
                au = activations[i][u]
                # if au == 0:
                #     continue
                for v in dst_nodes:
                    grads[u, v] += delta[v] * au
            if i > 0:
                # backpropagate delta to previous layer (pre-activation grads already folded into relu deriv)
                new_delta = np.zeros(len(activations[i]))
                for u in src_nodes:
                    s = 0.0
                    for v in dst_nodes:
                        s += self.W[u, v] * delta[v]
                    new_delta[u] = s * self._relu_deriv(activations[i][u])
                delta = new_delta
        # SGD update
        # print(f"[dbg] ||delta_out||={np.linalg.norm(delta):.3e}  ||grads||={np.linalg.norm(grads):.3e}")

        self.W -= self.learning_rate * grads

    # --- public API ---
    def train(self, data):
        for x_input, y_target in data:
            x0 = np.zeros(self.total_nodes)
            x0[self.layer_indices[0]] = x_input
            y_full = np.zeros(self.total_nodes)
            y_full[self.layer_indices[-1]] = y_target
            activations = self.forward(x0)
            self.backward(activations, y_full)

    def test(self, x_input):
        x0 = np.zeros(self.total_nodes)
        x0[self.layer_indices[0]] = x_input
        activations = self.forward(x0)
        return activations[-1][self.layer_indices[-1]]

    def save(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump({'n': self.n, 'W': self.W}, f)

    def load(self, filepath):
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
            self.n = state['n']
            self.layer_indices = self._get_default_layer_indices(self.n)
            self.total_nodes = sum(len(l) for l in self.layer_indices)
            self.W = state['W']

    def _graphW(self, activation_only=True, detail=""):
        title = "Activation Structure" if activation_only else "Weight Matrix"
        filename = f"figures/hexnet_n{self.n}_{title.replace(' ', '_')}{'_' + detail if detail else ''}.png"
        matrix = (self.W != 0).astype(int) if activation_only else self.W
        plt.figure(figsize=(7, 7))
        plt.imshow(matrix, cmap='Greys' if activation_only else 'viridis', interpolation='none')
        plt.suptitle(title + f" (n={self.n})")
        plt.title(f"lr={self.learning_rate}, {detail}")
        plt.xticks(np.arange(self.total_nodes))
        plt.yticks(np.arange(self.total_nodes))
        plt.grid(visible=True, color='black', linewidth=0.5)
        plt.colorbar()
        plt.savefig(filename)
        plt.show()

    # --- animated training ---
    def train_animated(self, data, epochs=25, classification=True, threshold=0.5, pause=0.05):
        """
        Train while animating loss & accuracy over epochs.
        - data: iterable of (x_input, y_target) with shapes (n,) and (n,)
        - classification=True: accuracy computed by thresholding outputs at `threshold`
          else we report 'pseudo-accuracy' as 1 / (1 + MAE) for a rough regression score.
        """
        losses = []
        accs = []

        # Prepare two separate figures to respect "one chart per figure"
        fig_loss = plt.figure(figsize=(6, 4))
        ax_loss = fig_loss.add_subplot(111)
        line_loss, = ax_loss.plot([], [])
        ax_loss.set_title("Training Loss (MAE)")
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

        # training loop
        for epoch in range(epochs):
            total_loss = 0.0
            correct = 0
            count = 0

            for x_input, y_target in data:
                # build padded vectors
                x0 = np.zeros(self.total_nodes)
                x0[self.layer_indices[0]] = x_input
                y_full = np.zeros(self.total_nodes)
                y_full[self.layer_indices[-1]] = y_target
                # print("--------------------------------")

                # print(f"x input={x_input}, x_padded={x0}, y expected={y_target}, y_padded={y_full}")

                activations = self.forward(x0)
                # print(f"activations={activations}")

                # for li, a in enumerate(activations):
                #     pos = np.count_nonzero(a > 0)
                #     neg = np.count_nonzero(a < 0)
                #     total = len(a)
                #     if li < len(self.layer_indices) - 1:
                #         print(f"layer {li}: pos={pos}, neg={neg}, total={total}, ratio={pos/total}")

                # loss (MSE on final layer nodes only)
                y_pred = activations[-1][self.layer_indices[-1]]
                diff = y_pred - y_target
                delta_thr = 0.1  # tune: try 0.1, 0.25, 1.0
                huber = np.where(np.abs(diff) <= delta_thr,
                                0.5 * diff**2,
                                delta_thr * (np.abs(diff) - 0.5 * delta_thr))
                total_loss += np.mean(huber)

                # # accuracy
                # if classification:
                #     pred_bin = (y_pred >= threshold).astype(int)
                #     tgt_bin = (y_target >= 0.5).astype(int)
                #     correct += np.mean(pred_bin == tgt_bin)
                # else:
                # a simple bounded score for regression feel: 1/(1+MAE)
                mae = np.mean(np.abs(y_pred - y_target))
                correct += 1.0 / (1.0 + mae)

                count += 1

                # backprop step
                self.backward(activations, y_full)

                # if epoch % 10 == 0:
                #     activations_after = self.forward(x0)
                #     y_pred_after = activations_after[-1][self.layer_indices[-1]]
                #     mae_after = np.mean(np.abs(y_pred_after - y_target))
                #     print(f"MAE before: {mae}, after: {mae_after}, delta_y={np.linalg.norm(y_pred_after - y_pred)}")

            epoch_loss = total_loss
            epoch_acc = correct / max(count, 1.0)
            losses.append(epoch_loss)
            accs.append(epoch_acc)

            # update plots
            line_loss.set_data(np.arange(1, len(losses)+1), losses)
            ax_loss.relim()
            ax_loss.autoscale_view()
            fig_loss.canvas.draw()

            line_acc.set_data(np.arange(1, len(accs)+1), accs)
            ax_acc.relim()
            ax_acc.autoscale_view()
            fig_acc.canvas.draw()
            plt.pause(pause)

            if epoch == epochs - 1:
                fig_loss.savefig(f"figures/hexnet_n{self.n}_loss_{epoch + 1}.png")
                fig_acc.savefig(f"figures/hexnet_n{self.n}_acc_{epoch + 1}.png")

        return np.array(losses), np.array(accs)


# ---------- Demo with synthetic data ----------
# We'll create a tiny binary task on n=2 (output matches input for half the samples)
n = 2
net = HexagonalNeuralNetwork(n=n, random_init=True, lr=0.01)

# # simple dataset: inputs in {0,1}^n and targets = inputs (identity) for demo
train_samples = 100
X = (np.random.rand(train_samples, n) * 2 - 1).astype(float)
Y = X.copy()
data = list(zip(X, Y))

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

net.graph(activation_only=False, detail='untrained')

# run animation
losses, accs = net.train_animated(data, epochs=100, pause=0.05)

# Show final structure & weights for sanity
# net.graph(activation_only=True)
net.graph(activation_only=False, detail='trained')