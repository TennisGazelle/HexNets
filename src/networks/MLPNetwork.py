import pickle
from typing import List
import matplotlib.pyplot as plt
import numpy as np
import pathlib

from src.networks.network import BaseNeuralNetwork
from src.networks.activation.activations import BaseActivation
from src.networks.loss.loss import BaseLoss
from src.networks.activation.Sigmoid import Sigmoid
from src.networks.loss.MeanSquaredErrorLoss import MeanSquaredErrorLoss
from src.networks.activation.activations import get_activation_function
from src.networks.loss.loss import get_loss_function


class MLPNetwork(BaseNeuralNetwork, display_name="mlp"):
    def __init__(
        self,
        input_dim: int = 3,
        output_dim: int = 3,
        hidden_dims: List[int] = [3],
        learning_rate: float = 0.01,
        activation: BaseActivation = Sigmoid,
        loss: BaseLoss = MeanSquaredErrorLoss,
    ):
        super().__init__(learning_rate, activation, loss)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.epochs_completed = 0
        self.W = []
        self.delta_W = []
        self.training_metrics = self.init_training_metrics()
        self.epochs_completed = 0

        if len(hidden_dims) == 0:
            raise ValueError("hidden_dims must be a list of at least one integer")

        # add layers
        self._add_layer(input_dim, hidden_dims[0])
        for i in range(1, len(hidden_dims)):
            self._add_layer(hidden_dims[i - 1], hidden_dims[i])
        self._add_layer(hidden_dims[-1], output_dim)

    # --- structure helpers ---
    def _add_layer(self, incoming_dim: int, outgoing_dim: int):
        weights = np.random.random((incoming_dim, outgoing_dim)) - 0.5
        self.W.append(weights)
        self.delta_W.append(np.zeros((incoming_dim, outgoing_dim)))

    def save(self, filepath) -> None:
        with open(filepath, "wb") as f:
            pickle.dump(
                {
                    "input_dim": self.input_dim,
                    "output_dim": self.output_dim,
                    "hidden_dims": self.hidden_dims,
                    "weights": self.W,
                    "training_metrics": self.training_metrics,
                    "learning_rate": self.learning_rate,
                    "activation": self.activation.display_name,
                    "loss": self.loss.display_name,
                    "epochs_completed": self.epochs_completed,
                },
                f,
            )
        print(f"Model saved to: {filepath}")

    def load(self, filepath):
        with open(filepath, "rb") as f:
            data = pickle.load(f)
            self.input_dim = data["input_dim"]
            self.output_dim = data["output_dim"]
            self.hidden_dims = data["hidden_dims"]
            self.W = data["weights"]
            self.training_metrics = data["training_metrics"]
            self.learning_rate = data["learning_rate"]
            self.activation = get_activation_function(data["activation"])
            self.loss = get_loss_function(data["loss"])
            self.epochs_completed = data["epochs_completed"]
            print(f"Model loaded from: {filepath}")

    def forward(self, x: np.ndarray) -> np.ndarray:
        activations = [x.copy()]
        a = x.copy()
        for i in range(len(self.W)):
            z = self.W[i].T @ a
            if i < len(self.W) - 1:
                a = self.activation.activate(z)
            else:
                a = z
            # a = self.activation.activate(z)
            activations.append(a.copy())
        return activations

    def backward(self, activations: np.ndarray, target: np.ndarray, apply_delta_W: bool = True):
        grads = [np.zeros_like(w) for w in self.W]
        delta = self.loss.calc_delta(target, self.activation.deactivate(activations[-1]))

        for i in reversed(range(len(self.W))):
            grads[i] = activations[i + 1] @ delta

            if i > 0:
                delta = delta @ self.W[i].T * self.activation.deactivate(activations[i])

        if apply_delta_W:
            for i in range(len(self.W)):
                self.delta_W[i] += self.learning_rate * grads[i]
        else:
            for i in range(len(self.W)):
                self.W[i] -= self.learning_rate * grads[i]

    def apply_delta_W(self):
        for i in range(len(self.W)):
            self.W[i] -= self.learning_rate * self.delta_W[i]
            self.delta_W[i].fill(0)

    def train(self, data):
        pass

    def test(self, x):
        pass

    def show_stats(self):
        print(f"MLP Network Stats:")
        print(f"layer_sizes:\t{[self.input_dim] + self.hidden_dims + [self.output_dim]}")
        print(f"lr:\t{self.learning_rate:.3f}")
        print(f"epochs completed:\t{self.epochs_completed}")
        print(f"loss_method:\t{self.loss.display_name}")
        print(f"activation_method:\t{self.activation.display_name}")

        # print(f"loss:\t{self.training_metrics['loss'][-1]:.3f}")
        # print(f"accuracy:\t{self.training_metrics['accuracy'][-1]:.3f}")
        # print(f"r_squared:\t{self.training_metrics['r_squared'][-1]:.3f}")

    def graph_weights(self, activation_only=True, detail="", output_dir: pathlib.Path = None):
        pass

    def graph_structure(self, detail="", output_dir: pathlib.Path = None):
        pass

    def train_animated(
        self, data, epochs=25, pause=0.05, output_dir: pathlib.Path = None
    ) -> tuple[float, float, float]:
        """
        Train while animating loss & accuracy over epochs.
        - data: iterable of (x_input, y_target) with shapes (n,) and (n,)
        """
        print(f"MLP Network Training:")
        print(f"epochs:\t{epochs}")
        print(f"datapoints:\t{len(data)}")

        print("Training...")
        parent_dir = output_dir if output_dir else pathlib.Path("figures")
        parent_dir.mkdir(parents=True, exist_ok=True)

        # three charts in a single figure
        fig, (ax_loss, ax_acc, ax_r2) = plt.subplots(3, 1, figsize=(6, 12))
        fig.suptitle(f"Training {self.display_name} ({self.loss}, {self.activation})")

        (line_loss,) = ax_loss.plot([], [])
        ax_loss.set_title(f"Loss ({self.loss})")
        # ax_loss.set_xlabel("Epoch")
        ax_loss.set_ylabel("Loss")
        ax_loss.grid(True)

        (line_acc,) = ax_acc.plot([], [])
        ax_acc.set_title(f"Accuracy (RMSE)")
        # ax_acc.set_xlabel("Epoch")
        ax_acc.set_ylabel("Accuracy")
        ax_acc.set_ylim(0, 1)
        ax_acc.grid(True)

        (line_r2,) = ax_r2.plot([], [])
        ax_r2.set_title(f"R^2 (coefficient of determination)")
        ax_r2.set_xlabel("Epoch")
        ax_r2.set_ylabel("R^2")
        ax_r2.grid(True)

        # training loop
        for epoch in range(self.epochs_completed, epochs + self.epochs_completed):
            total_loss = 0.0
            correct = 0
            count = 0
            ss_res_sum = 0.0
            sum_y = 0.0
            sum_y2 = 0.0
            num_elems = 0

            for x_input, y_target in data:
                # print('---')
                activations = self.forward(x_input)

                # for li, a in enumerate(activations):
                #     pos = np.count_nonzero(a > 0)
                #     neg = np.count_nonzero(a < 0)
                #     zeros = np.count_nonzero(a == 0)
                #     total = len(a)
                #     print(f"layer {li}: {a}")
                #     print(f"layer {li}: pos={pos}, neg={neg}, zeros={zeros}, total={total}, ratio={pos/total}")

                # loss (MSE on final layer nodes only)
                y_pred = activations[-1]
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

                rmse = np.sqrt(np.mean((y_pred - y_target) ** 2))
                score = np.exp(-rmse)  # maps 0->1, larger errors decay smoothly
                correct += score

                # scale = np.maximum(1e-8, np.mean(np.abs(y_target)))  # per-sample
                # nmae = np.mean(np.abs(y_pred - y_target)) / scale
                # correct += np.exp(-nmae)

                # r_squared
                ss_res_sum += float(np.sum((y_pred - y_target) ** 2))
                sum_y += float(np.sum(y_target))
                sum_y2 += float(np.sum(y_target**2))

                count += 1
                num_elems += y_pred.shape[0]

                # backprop step
                self.backward(activations, y_target, apply_delta_W=False)

                # if epoch % 10 == 0:
                #     activations_after = self.forward(x0)
                #     y_pred_after = activations_after[-1][self.layer_indices[-1]]
                #     mae_after = np.mean(np.abs(y_pred_after - y_target))
                #     print(f"MAE before: {mae}, after: {mae_after}, delta_y={np.linalg.norm(y_pred_after - y_pred)}")

            # r_squared, cont'd
            ss_tot = sum_y2 - (sum_y**2 / (num_elems))
            r_squared = 1 - (ss_res_sum / (ss_tot + 1e-12))

            epoch_loss = total_loss / count
            epoch_acc = correct / count
            epoch_r2 = r_squared
            self.training_metrics["loss"].append(epoch_loss)
            self.training_metrics["accuracy"].append(epoch_acc)
            self.training_metrics["r_squared"].append(epoch_r2)
            self.apply_delta_W()
            self.epochs_completed += 1

            # update plots
            line_loss.set_data(
                np.arange(1, len(self.training_metrics["loss"]) + 1),
                self.training_metrics["loss"],
            )
            ax_loss.relim()
            ax_loss.autoscale_view()
            # fig_loss.canvas.draw()

            line_acc.set_data(
                np.arange(1, len(self.training_metrics["accuracy"]) + 1),
                self.training_metrics["accuracy"],
            )
            ax_acc.relim()
            ax_acc.autoscale_view()
            # fig_acc.canvas.draw()

            line_r2.set_data(
                np.arange(1, len(self.training_metrics["r_squared"]) + 1),
                self.training_metrics["r_squared"],
            )
            ax_r2.relim()
            ax_r2.autoscale_view()
            # fig_r2.canvas.draw()

            fig.canvas.draw()

            plt.pause(pause)

            if epoch == epochs + self.epochs_completed - 1:
                training_figure_filename = f"mlp_training_{self.loss}_{self.activation}_{epoch + 1}.png"
                # fig_loss_filename = f"figures/hexnet_n{self.n}_r{self.r}_loss-{self.loss}_{epoch + 1}.png"
                # fig_acc_filename = f"figures/hexnet_n{self.n}_r{self.r}_acc_{epoch + 1}.png"
                # fig_r2_filename = f"figures/hexnet_n{self.n}_r{self.r}_r2_{epoch + 1}.png"
                # fig_loss.savefig(fig_loss_filename)
                # fig_acc.savefig(fig_acc_filename)
                # fig_r2.savefig(fig_r2_filename)
                plt.savefig(parent_dir / training_figure_filename)
                print("")
                print(f"Training complete!")
                print(f"Loss: \t\t {epoch_loss:.3f}")
                print(f"Accu: \t\t {epoch_acc:.3f}")
                print(f"R^2: \t\t {epoch_r2:.3f}")
                print(f"Epochs completed: {epoch + 1}")
                print(f"Training figure saved to: {parent_dir / training_figure_filename}")
                # print(f"Loss output: \t {fig_loss_filename}")
                # print(f"Accu output: \t {fig_acc_filename}")
                # print(f"R^2 output: \t {fig_r2_filename}")

        return epoch_loss, epoch_acc, epoch_r2

    def get_metrics_json(self):
        return self.training_metrics
