import os
import pickle
from typing import List

import matplotlib.pyplot as plt
import numpy as np

from src.networks.network import BaseNeuralNetwork
from src.networks.activations import LeakyReLU, ReLU, Sigmoid
from src.networks.loss import HuberLoss, LogCoshLoss, MeanSquaredError, QuantileLoss

# === Hexagonal Neural Network ===
class HexagonalNeuralNetwork(BaseNeuralNetwork):
    def __init__(self, n, r=0, random_init=True, lr=0.01):
        super().__init__(n, Sigmoid(), MeanSquaredError())
        self.total_nodes = self._calc_total_nodes(n)
        self.dir_metrics = {
            i: { 
                'W': self._init_weight_matrix(i, random_init=random_init), 
                'indices': self._get_layer_indices(n, r=i)
            } for i in range(0, 6)
        }
        self.r = r
        self.learning_rate = lr
        self.training_metrics = {
            "loss": [],
            "accuracy": [],
            "r_squared": []
        }

    # --- structure helpers ---
    def _calc_total_nodes(self, n):
        return sum(l for l in self._hex_layer_sizes(n))

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
    
    def _get_layer_indices(self, n, r=1):
        assert 0 <= r <= 5, f"Invalid rotation: {r}"
        max_row_size = 2*n - 1
        
        def _get_top_down():
            return self._get_default_layer_indices(n)

        if r == 0:
            return _get_top_down()
        if r == 3:
            return [rw[::-1] for rw in _get_top_down()[::-1]]

        def _get_top_right_to_bottom_left():
            new_indices = [[] for _ in range(max_row_size)]
            node_idx = 0
            for row_idx in range(max_row_size):
                min_new_row_idx = max(0, (row_idx - n + 1))
                max_new_row_idx = min(max_row_size - 1, row_idx + n - 1)
                for new_row_idx in range(min_new_row_idx, max_new_row_idx + 1):
                    new_indices[new_row_idx].append(node_idx)
                    node_idx += 1
            return new_indices

        if r == 1:
            return [rw[::-1] for rw in _get_top_right_to_bottom_left()]
        if r == 4:
            return [rw for rw in _get_top_right_to_bottom_left()[::-1]]
        
        def _get_top_left_to_bottom_right():
            new_indices = [[] for _ in range(max_row_size)]
            node_idx = 0
            for row_idx in range(max_row_size):
                min_new_row_idx = max(0, (row_idx - n + 1))
                max_new_row_idx = min(max_row_size - 1, row_idx + n - 1)
                for new_row_idx in range(max_new_row_idx, min_new_row_idx - 1, -1):
                    new_indices[new_row_idx].append(node_idx)
                    node_idx += 1
            return new_indices
        
        if r == 5:
            return [rw for rw in _get_top_left_to_bottom_right()]
        if r == 2:
            return [rw[::-1] for rw in _get_top_left_to_bottom_right()[::-1]]
        
        raise ValueError(f"Invalid rotation: {r}")

    # --- weights ---
    def _init_weight_matrix(self, this_r: int, random_init=True):
        W = np.zeros((self.total_nodes, self.total_nodes))
        layer_indices = self._get_layer_indices(self.n, r=this_r)
        count = 1
        for i in range(len(layer_indices) - 1):
            for u in layer_indices[i]:
                for v in layer_indices[i + 1]:
                    W[u, v] = np.random.randn() if random_init else count
                    count += 1
        return W

    # --- forward & backward ---
    def pad_input(self, x, this_r: int = 0):
        if this_r == 0:
            x0 = np.zeros(self.total_nodes)
            x0[self.dir_metrics[this_r]['indices'][0]] = x
            return x0
        else:
            x0 = np.zeros(self.total_nodes)
            # todo: implement this padding for each r
            return x0

    def unpad_output(self, y, this_r: int = 0):
        if this_r == 0:
            return y[self.dir_metrics[this_r]['indices'][-1]]
        else:
            return y[self.dir_metrics[this_r]['indices'][-1]]

    def forward(self, x):
        activations = [x.copy()]
        a = x.copy()
        for i in range(len(self.dir_metrics[self.r]['indices']) - 1):
            z = self.dir_metrics[self.r]['W'].T @ a
            if i < len(self.dir_metrics[self.r]['indices']) - 2:
                a = self.activation.activate(z)
            else:
                a = z
            activations.append(a.copy())
        return activations

    def backward(self, activations, target):
        grads = np.zeros_like(self.dir_metrics[self.r]['W'])
        delta = self.loss.calc_delta(target, activations[-1])
        # walk layers backward
        for i in reversed(range(len(self.dir_metrics[self.r]['indices']) - 1)):
            src_nodes = self.dir_metrics[self.r]['indices'][i]
            dst_nodes = self.dir_metrics[self.r]['indices'][i + 1]
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
                        s += self.dir_metrics[self.r]['W'][u, v] * delta[v]
                    new_delta[u] = s * self.activation.deactivate(activations[i][u])
                delta = new_delta
        # SGD update
        # print(f"[dbg] ||delta_out||={np.linalg.norm(delta):.3e}  ||grads||={np.linalg.norm(grads):.3e}")

        self.dir_metrics[self.r]['W'] -= self.learning_rate * grads

    # --- public API ---
    def train(self, data):
        for x_input, y_target in data:
            x_full = self.pad_input(x_input, self.r)
            y_full = np.zeros(self.total_nodes)
            y_full[self.dir_metrics[self.r]['indices'][-1]] = y_target
            activations = self.forward(x_full)
            self.backward(activations, y_full)

    def test(self, x_input):
        x_full = self.pad_input(x_input, self.r)
        activations = self.forward(x_full)
        return self.unpad_output(activations[-1], self.r)

    def save(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump({
                'n': self.n, 
                'dir_metrics': self.dir_metrics, 
                'training_metrics': self.training_metrics
            }, f)

    def load(self, filepath):
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
            self.n = state['n']
            self.total_nodes = self._calc_total_nodes(self.n)
            self.dir_metrics = {
                i: { 
                    'W': state['dir_metrics'][i]['W'], 
                    'indices': state['dir_metrics'][i]['indices']
                } for i in range(0, 6)
            }
            self.training_metrics = state['training_metrics']

    def _graphW(self, activation_only=True, detail=""):
        title = "Activation Structure" if activation_only else "Weight Matrix"
        filename = f"figures/hexnet_n{self.n}_r{self.r}_{title.replace(' ', '_')}{'_' + detail if detail else ''}.png"
        matrix = (self.dir_metrics[self.r]['W'] != 0).astype(int) if activation_only else self.dir_metrics[self.r]['W']

        plt.figure(figsize=(7, 7))
        plt.imshow(matrix, cmap='Greys' if activation_only else 'viridis', interpolation='none')
        plt.suptitle(title + f" (n={self.n}, r={self.r})")
        plt.title(f"lr={self.learning_rate}, {detail}")
        plt.xticks(np.arange(self.total_nodes))
        plt.yticks(np.arange(self.total_nodes))
        # plt.grid(visible=True, color='black', linewidth=0.5)
        plt.colorbar()
        plt.savefig(filename)
        plt.show()

        return filename
    
    def _printIndices(self, r):
        for i, layer in enumerate(self.dir_metrics[r]['indices']):
            print(f"layer{i}: {layer}")

    def _graph_hex(
        self,
        figsize=(10, 10),
        node_radius=0.28,
        edge_alpha=1,
        edge_weighted=True,          # if True, scale edge alpha by |W[u,v]|
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
        layers = self.dir_metrics[self.r]['indices']  # assumes you've already computed this
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

        # # --- edge weight normalization (if using W) ---
        # def edge_alpha_for(u, v):
        #     if not edge_weighted or getattr(self, "W", None) is None:
        #         return edge_alpha
        #     w = abs(self.W[u, v])
        #     # robust normalization: divide by (median(abs(W_next_layer)) + tiny epsilon)
        #     # so outliers don't blow up visualization
        #     return min(1.0, edge_alpha * (w / (edge_norms.get(i, 1e-9))))

        # edge_norms = {}
        # if edge_weighted and getattr(self, "W", None) is not None:
        #     for i in range(len(layers) - 1):
        #         u_nodes = layers[i]
        #         v_nodes = layers[i + 1]
        #         block = np.abs(self.W[np.ix_(u_nodes, v_nodes)])
        #         med = np.median(block) if block.size else 1.0
        #         edge_norms[i] = med if med > 0 else (np.max(block) if np.max(block) > 0 else 1.0)

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
                        color="black",
                        zorder=1,
                        # alpha=edge_alpha_for(u, v),
                    )

        # nodes
        for node, (x, y) in pos.items():
            circ = plt.Circle((x, y), node_radius, facecolor=node_fc, edgecolor=node_ec, linewidth=1.2, zorder=10)
            ax.add_patch(circ)
            if label == "value" and values is not None:
                txt = f"{values[node]:.3g}" if isinstance(values[node], (int, float, np.floating)) else str(values[node])
            else:
                txt = str(node)
            ax.text(x, y, txt, ha="center", va="center", fontsize=font_size, zorder=11)

        ax.set_aspect("equal")
        ax.axis("off")

        # tidy margins
        xs, ys = zip(*pos.values())
        pad = 1.5 * node_radius + 0.2
        ax.set_xlim(min(xs) - pad, max(xs) + pad)
        ax.set_ylim(min(ys) - pad, max(ys) + pad)

        plt.tight_layout()
        filename = f"figures/hexnet_n{self.n}_r{self.r}_view.png"
        plt.savefig(filename)
        plt.show()

        return filename


    def to_dot_string(self) -> List[str]:
        """
        Export the layered, fully-connected structure as Graphviz DOT text.
        Node IDs are your global indices. Layers are grouped with ranks.
        """
        lines = []
        lines.append("digraph HexNet {")
        lines.append('  graph [rankdir=TB, splines=false, nodesep="0.35", ranksep="0.35"];')
        lines.append('  node  [shape=circle, fontsize=10, width=0.45, fixedsize=true, zorder=10];')
        lines.append('  edge  [penwidth=0.5, dir="none", zorder=1];')

        # group nodes by layer (same rank)
        for i, layer in enumerate(self.dir_metrics[self.r]['indices']):
            lines.append(f'  {{ rank=same; // layer {i}')
            for n in layer:
                lines.append(f'    {n};')
            lines.append("  }")
        
        # invisible edges between nodes in same rank to enforce order
        for i in range(len(self.dir_metrics[self.r]['indices'])):
            line = f"  {self.dir_metrics[self.r]['indices'][i][0]}"
            for u_idx, u in enumerate(self.dir_metrics[self.r]['indices'][i]):
                if u_idx > 0:
                    line += f" -> {u}"
            line += " [style=\"invis\"];"
            lines.append(line)

        # edges between consecutive layers
        for i in range(len(self.dir_metrics[self.r]['indices']) - 1):
            for u in self.dir_metrics[self.r]['indices'][i]:
                for v in self.dir_metrics[self.r]['indices'][i + 1]:
                    if getattr(self, "W", None) is not None and False:
                        w = float(self.dir_metrics[self.r]['W'][u, v])
                        # include a lightweight weight attribute (optional)
                        lines.append(f'  {u} -> {v} [penwidth=1.0, weight="{w:.3g}"];')
                    else:
                        lines.append(f"  {u} -> {v};")

        lines.append("}")
        return lines

    def _graph_hex_dot(self):
        dot_string_list = self.to_dot_string()
        dot_file = f"figures/hexnet_n{self.n}_r{self.r}_viewdot.dot"
        with open(dot_file, 'w') as f:
            for line in dot_string_list:
                f.write(line + "\n")

        png_file = dot_file.replace('.dot', '.png')
        os.system(f"dot -Tpng {dot_file} -o {png_file}")

        return png_file

    # --- animated training ---
    def train_animated(self, data, epochs=25, threshold=0.5, pause=0.05) -> tuple[float, float, float]:
        """
        Train while animating loss & accuracy over epochs.
        - data: iterable of (x_input, y_target) with shapes (n,) and (n,)
        """
        print(f"Network:")
        print(f"n:\t{self.n}")
        print(f"lr:\t{self.learning_rate}")
        print(f"epochs:\t{epochs}")
        print(f"loss:\t{self.loss.name}")
        print(f"activation:\t{self.activation.name}")
        print("Training...")

        # Prepare separate figures to respect "one chart per figure"
        fig_loss = plt.figure(figsize=(6, 4))
        ax_loss = fig_loss.add_subplot(111)
        line_loss, = ax_loss.plot([], [])
        ax_loss.set_title(f"Training Loss ({self.loss.name})")
        ax_loss.set_xlabel("Epoch")
        ax_loss.set_ylabel("Loss")
        ax_loss.grid(True)

        fig_acc = plt.figure(figsize=(6, 4))
        ax_acc = fig_acc.add_subplot(111)
        line_acc, = ax_acc.plot([], [])
        ax_acc.set_title(f"Training Accuracy")
        ax_acc.set_xlabel("Epoch")
        ax_acc.set_ylabel("Accuracy")
        ax_acc.set_ylim(0, 1)
        ax_acc.grid(True)

        fig_r2 = plt.figure(figsize=(6, 4))
        ax_r2 = fig_r2.add_subplot(111)
        line_r2, = ax_r2.plot([], [])
        ax_r2.set_title(f"Training R^2")
        ax_r2.set_xlabel("Epoch")
        ax_r2.set_ylabel("R^2")
        ax_r2.grid(True)

        # training loop
        for epoch in range(epochs):
            total_loss = 0.0
            correct = 0
            count = 0
            ss_res_sum = 0.0
            sum_y = 0.0
            sum_y2 = 0.0

            for x_input, y_target in data:
                # build padded vectors
                x_full = self.pad_input(x_input, self.r)
                y_full = np.zeros(self.total_nodes)
                y_full[self.dir_metrics[self.r]['indices'][-1]] = y_target
                # print("--------------------------------")

                # print(f"x input={x_input}, x_padded={x0}, y expected={y_target}, y_padded={y_full}")

                activations = self.forward(x_full)
                # print(f"activations={activations}")

                # for li, a in enumerate(activations):
                #     pos = np.count_nonzero(a > 0)
                #     neg = np.count_nonzero(a < 0)
                #     total = len(a)
                #     if li < len(self.layer_indices) - 1:
                #         print(f"layer {li}: pos={pos}, neg={neg}, total={total}, ratio={pos/total}")

                # loss (MSE on final layer nodes only)
                y_pred = self.unpad_output(activations[-1], self.r)
                total_loss += self.loss.calc_loss(y_target, y_pred)

                # accuracy
                # if classification:
                #     pred_bin = (y_pred >= threshold).astype(int)
                #     tgt_bin = (y_target >= 0.5).astype(int)
                #     correct += np.mean(pred_bin == tgt_bin)
                # else:
                # a simple bounded score for regression feel: 1/(1+MAE)
                # mae = np.mean(np.abs(y_pred - y_target))
                # correct += 1.0 / (1.0 + mae)

                rmse = np.sqrt(np.mean((y_pred - y_target)**2))
                score = np.exp(-rmse)  # maps 0->1, larger errors decay smoothly
                correct += score

                # scale = np.maximum(1e-8, np.mean(np.abs(y_target)))  # per-sample
                # nmae = np.mean(np.abs(y_pred - y_target)) / scale
                # correct += np.exp(-nmae)

                # r_squared
                ss_res_sum += float(np.sum((y_pred - y_target)**2))
                sum_y += float(np.sum(y_target))
                sum_y2 += float(np.sum(y_target**2))

                count += 1

                # backprop step
                self.backward(activations, y_full)

                # if epoch % 10 == 0:
                #     activations_after = self.forward(x0)
                #     y_pred_after = activations_after[-1][self.layer_indices[-1]]
                #     mae_after = np.mean(np.abs(y_pred_after - y_target))
                #     print(f"MAE before: {mae}, after: {mae_after}, delta_y={np.linalg.norm(y_pred_after - y_pred)}")

            # r_squared, cont'd
            ss_tot = sum_y2 - (sum_y**2 / count)
            r_squared = 1 - (ss_res_sum / (ss_tot + 1e-12))

            epoch_loss = total_loss / count
            epoch_acc = correct / count
            epoch_r2 = r_squared
            self.training_metrics["loss"].append(epoch_loss)
            self.training_metrics["accuracy"].append(epoch_acc)
            self.training_metrics["r_squared"].append(epoch_r2)

            # update plots
            line_loss.set_data(np.arange(1, len(self.training_metrics["loss"])+1), self.training_metrics["loss"])
            ax_loss.relim()
            ax_loss.autoscale_view()
            fig_loss.canvas.draw()

            line_acc.set_data(np.arange(1, len(self.training_metrics["accuracy"])+1), self.training_metrics["accuracy"])
            ax_acc.relim()
            ax_acc.autoscale_view()
            fig_acc.canvas.draw()

            line_r2.set_data(np.arange(1, len(self.training_metrics["r_squared"])+1), self.training_metrics["r_squared"])
            ax_r2.relim()
            ax_r2.autoscale_view()
            fig_r2.canvas.draw()

            plt.pause(pause)

            if epoch == epochs - 1:
                fig_loss_filename = f"figures/hexnet_n{self.n}_r{self.r}_loss-{self.loss.name}_{epoch + 1}.png"
                fig_acc_filename = f"figures/hexnet_n{self.n}_r{self.r}_acc_{epoch + 1}.png"
                fig_r2_filename = f"figures/hexnet_n{self.n}_r{self.r}_r2_{epoch + 1}.png"
                fig_loss.savefig(fig_loss_filename)
                fig_acc.savefig(fig_acc_filename)
                fig_r2.savefig(fig_r2_filename)
                print("")
                print(f"Training complete!")
                print(f"Loss: \t\t {epoch_loss:.3f}")
                print(f"Accu: \t\t {epoch_acc:.3f}")
                print(f"R^2: \t\t {epoch_r2:.3f}")
                print(f"Loss output: \t {fig_loss_filename}")
                print(f"Accu output: \t {fig_acc_filename}")
                print(f"R^2 output: \t {fig_r2_filename}")

        return epoch_loss, epoch_acc, epoch_r2
