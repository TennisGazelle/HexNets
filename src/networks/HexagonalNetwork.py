import os
import pickle

import matplotlib.pyplot as plt
import numpy as np

from src.networks.network import BaseNeuralNetwork

# === Hexagonal Neural Network ===
class HexagonalNeuralNetwork(BaseNeuralNetwork):
    def __init__(self, n, random_init=True, lr=0.01):
        self.n = n
        self.layer_indices = self._get_default_layer_indices(n)
        self.total_nodes = sum(len(l) for l in self.layer_indices)
        self.W = self._init_weight_matrix(random_init=random_init)
        self.learning_rate = lr
        self.training_metrics = {
            "loss": [],
            "accuracy": []
        }

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
                    W[u, v] = np.random.randn() if random_init else 0.0
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
            pickle.dump({'n': self.n, 'W': self.W, 'training_metrics': self.training_metrics}, f)

    def load(self, filepath):
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
            self.n = state['n']
            self.layer_indices = self._get_default_layer_indices(self.n)
            self.total_nodes = sum(len(l) for l in self.layer_indices)
            self.W = state['W']
            self.training_metrics = state['training_metrics']

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

        return filename
    
    def _printIndices(self):
        for i, layer in enumerate(self.layer_indices):
            print(f"layer{i}: {layer}")

    def graphHex(
        self,
        figsize=(10, 8),
        node_radius=0.28,
        edge_alpha=0.15,
        edge_weighted=False,          # if True, scale edge alpha by |W[u,v]|
        node_fc="white",
        node_ec="black",
        font_size=10,
        label="index",               # "index" (default) or "value"
        values=None,                 # optional array aligned to global node index
        dy=1.0,                      # vertical spacing
        dx=1.0                       # horizontal spacing
    ):
        """
        Draw a hex-layer network:
        - nodes are arranged in rows with sizes given by self.layer_indices
        - every node in layer i connects to every node in layer i+1
        """
        layers = self.layer_indices  # assumes you've already computed this
        if not layers:
            raise ValueError("layer_indices is empty. Initialize the network first.")

        # --- positions (centered rows, optional stagger for hex look) ---
        pos = {}  # node_id -> (x, y)
        for i, layer in enumerate(layers):
            size = len(layer)
            y = -i * dy
            # center the row; optionally stagger alternate rows by 0.5*dx
            offset = -0.5 * (size - 1) * dx
            for j, node in enumerate(layer):
                x = offset + j * dx
                pos[node] = (x, y)

        # --- edge weight normalization (if using W) ---
        def edge_alpha_for(u, v):
            if not edge_weighted or getattr(self, "W", None) is None:
                return edge_alpha
            w = abs(self.W[u, v])
            # robust normalization: divide by (median(abs(W_next_layer)) + tiny epsilon)
            # so outliers don't blow up visualization
            return min(1.0, edge_alpha * (w / (edge_norms.get(i, 1e-9))))

        edge_norms = {}
        if edge_weighted and getattr(self, "W", None) is not None:
            for i in range(len(layers) - 1):
                u_nodes = layers[i]
                v_nodes = layers[i + 1]
                block = np.abs(self.W[np.ix_(u_nodes, v_nodes)])
                med = np.median(block) if block.size else 1.0
                edge_norms[i] = med if med > 0 else (np.max(block) if np.max(block) > 0 else 1.0)

        # --- draw ---
        fig, ax = plt.subplots(figsize=figsize)

        # edges
        for i in range(len(layers) - 1):
            for u in layers[i]:
                x1, y1 = pos[u]
                for v in layers[i + 1]:
                    x2, y2 = pos[v]
                    ax.plot(
                        [x1, x2], [y1, y2],
                        linewidth=1.0,
                        alpha=edge_alpha_for(u, v),
                    )

        # nodes
        for node, (x, y) in pos.items():
            circ = plt.Circle((x, y), node_radius, facecolor=node_fc, edgecolor=node_ec, linewidth=1.2)
            ax.add_patch(circ)
            if label == "value" and values is not None:
                txt = f"{values[node]:.3g}" if isinstance(values[node], (int, float, np.floating)) else str(values[node])
            else:
                txt = str(node)
            ax.text(x, y, txt, ha="center", va="center", fontsize=font_size)

        ax.set_aspect("equal")
        ax.axis("off")

        # tidy margins
        xs, ys = zip(*pos.values())
        pad = 1.5 * node_radius + 0.2
        ax.set_xlim(min(xs) - pad, max(xs) + pad)
        ax.set_ylim(min(ys) - pad, max(ys) + pad)

        plt.tight_layout()
        filename = f"figures/hexnet_n{self.n}_view.png"
        plt.savefig(filename)
        plt.show()

        return filename


    def to_dot_string(self):
        """
        Export the layered, fully-connected structure as Graphviz DOT text.
        Node IDs are your global indices. Layers are grouped with ranks.
        """
        lines = []
        lines.append("digraph HexNet {")
        lines.append('  graph [rankdir=TB, splines=false, nodesep="0.35", ranksep="0.35"];')
        lines.append('  node  [shape=circle, fontsize=10, width=0.45, fixedsize=true];')
        lines.append('  edge  [penwidth=1.0, dir="none"];')

        # group nodes by layer (same rank)
        for i, layer in enumerate(self.layer_indices):
            lines.append(f'  {{ rank=same; // layer {i}')
            for n in layer:
                lines.append(f'    {n};')
            lines.append("  }")
        
        # invisible edges between nodes in same rank to enforce order
        for i in range(len(self.layer_indices)):
            line = f"  {self.layer_indices[i][0]}"
            for u_idx, u in enumerate(self.layer_indices[i]):
                if u_idx > 0:
                    line += f" -> {u}"
            line += " [style=\"invis\"];"
            lines.append(line)

        # edges between consecutive layers
        for i in range(len(self.layer_indices) - 1):
            for u in self.layer_indices[i]:
                for v in self.layer_indices[i + 1]:
                    if getattr(self, "W", None) is not None and False:
                        w = float(self.W[u, v])
                        # include a lightweight weight attribute (optional)
                        lines.append(f'  {u} -> {v} [penwidth=1.0, weight="{w:.3g}"];')
                    else:
                        lines.append(f"  {u} -> {v};")

        lines.append("}")
        return "\n".join(lines)

    def graph_dot(self):
        dot_string = self.to_dot_string()
        dot_file = f"figures/hexnet_n{self.n}_viewdot.dot"
        with open(dot_file, 'w') as f:
            f.write(dot_string)

        png_file = dot_file.replace('.dot', '.png')
        os.system(f"dot -Tpng {dot_file} -o {png_file}")

        return png_file

    # --- animated training ---
    def train_animated(self, data, epochs=25, classification=True, threshold=0.5, pause=0.05):
        """
        Train while animating loss & accuracy over epochs.
        - data: iterable of (x_input, y_target) with shapes (n,) and (n,)
        - classification=True: accuracy computed by thresholding outputs at `threshold`
          else we report 'pseudo-accuracy' as 1 / (1 + MAE) for a rough regression score.
        """
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
            self.training_metrics["loss"].append(epoch_loss)
            self.training_metrics["accuracy"].append(epoch_acc)

            # update plots
            line_loss.set_data(np.arange(1, len(self.training_metrics["loss"])+1), self.training_metrics["loss"])
            ax_loss.relim()
            ax_loss.autoscale_view()
            fig_loss.canvas.draw()

            line_acc.set_data(np.arange(1, len(self.training_metrics["accuracy"])+1), self.training_metrics["accuracy"])
            ax_acc.relim()
            ax_acc.autoscale_view()
            fig_acc.canvas.draw()
            plt.pause(pause)

            if epoch == epochs - 1:
                fig_loss_filename = f"figures/hexnet_n{self.n}_loss_{epoch + 1}.png"
                fig_acc_filename = f"figures/hexnet_n{self.n}_acc_{epoch + 1}.png"
                fig_loss.savefig(fig_loss_filename)
                fig_acc.savefig(fig_acc_filename)
                print(f"Training complete! Loss: {epoch_loss:.3f}, Accuracy: {epoch_acc:.3f}")
                print(f"Training Loss saved to {fig_loss_filename}")
                print(f"Training Accuracy saved to {fig_acc_filename}")

        return np.array(self.training_metrics["loss"]), np.array(self.training_metrics["accuracy"])
