import math
import random

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_derivative(x):
    return sigmoid(x) * sigmoid(1 - x)


def shoot_odds_one_in(odds):
    # Generate a random number between 1 and the odds value
    result = random.randint(1, odds)
    # Check if the result is equal to 1 (which simulates hitting the target)
    return result == 1


def get_loss_vec(target_vec, given_vec):
    return pow(target_vec - given_vec, 2)


class MLP:
    def __init__(self):
        self.weights = []  # List to store the weight matrices
        self.weights_delta = []  #

    def add_layer(self, input_size, output_size):
        # Randomly initialize weights for the layer
        weights = np.random.random((input_size, output_size)) - 0.5
        self.weights.append(weights)

        zeroes = np.zeros((input_size, output_size))
        self.weights_delta.append(zeroes)

    def feed_forward(self, X):
        output = X
        for layer_weights in self.weights:
            output = sigmoid(np.dot(output, layer_weights))
        return output

    def feed_forward_save_cache(self, X):
        output = X
        outputs = [X]
        for layer_weights in self.weights:
            output = sigmoid(np.dot(output, layer_weights))
            outputs.append(output)
        return outputs

    def get_cum_loss(self, X, y):
        overall_loss = 0
        overall_correct = 0
        buffer_size = 250
        for i in range(buffer_size):
            input_vec = X[i]
            target_vec = y[i]
            output_vec = self.feed_forward_save_cache(input_vec)[-1]

            loss = get_loss_vec(target_vec, output_vec)
            overall_loss += loss.sum()
            overall_correct += 1 if np.argmax(target_vec) == np.argmax(output_vec) else 0

        return overall_loss / buffer_size, overall_correct / buffer_size * 100

    def reverse_weights_if_possible(self):
        # todo, make this actually well done, just check for even right now
        if not len(self.weights) % 2 == 0:
            print("skipping reversing weights")
            return

        for i in range(len(self.weights)):
            self.weights[i] = self.weights[i].T
        self.weights.reverse()

    def train(self, X, y, num_epochs=1000, learning_rate=0.1, trauma_learning_rate=100.0):
        # print(f'input vecs({X.shape}): {X}')
        # print(f'output vecs({y.shape}): {y}')

        loss_values = []
        loss_percentage_change_values = []
        dynamic_learning_rates = []
        accuracy_values = []

        # Create a figure and axis for the plot
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, sharex="all", gridspec_kw={"height_ratios": [3, 1, 1, 2]})
        ax1.set(xlabel="Iteration", ylabel="Loss", xlim=num_epochs)
        ax2.set(ylabel="Learning Rate", xlim=num_epochs)
        ax3.set(ylabel="Delta Loss", xlim=num_epochs)
        ax4.set(ylabel="Accuracy on Average", xlim=num_epochs)

        ax1.grid()
        ax2.grid()
        ax3.grid()
        ax4.grid()

        for iteration in range(num_epochs):
            a = learning_rate
            # dynamic_learning_rate = learning_rate
            # dynamic_learning_rate = a * math.sin((math.pi/50)*(iteration+1)) + (a*1.5)
            # dynamic_learning_rate = a * math.sin((math.pi/1000) * (iteration+1)) * math.sin((math.pi/100) * (iteration+1)) + (a*1.5)
            dynamic_learning_rate = a / (iteration + 1) * math.sin((iteration + 1) * math.pi / 5) + (a * 1.5)

            self.train_epoch(X, y)

            # Graph stuff (non-important mathematically; not worth abstraction headache)
            numerical_loss, numerical_accuracy = self.get_cum_loss(X, y)
            percent_change = numerical_loss / (loss_values[-1] if loss_values else numerical_loss) - 1
            loss_values.append(numerical_loss)
            dynamic_learning_rates.append(dynamic_learning_rate)
            loss_percentage_change_values.append(percent_change)
            accuracy_values.append(numerical_accuracy)

            # Clear the axis and plot the updated values
            ax1.clear()
            ax2.clear()
            ax3.clear()
            ax4.clear()

            # ax.set_xlim(left=0, right=epochs)
            ax1.plot(loss_values)
            ax2.plot(loss_percentage_change_values)
            ax3.plot(dynamic_learning_rates)
            ax4.plot(accuracy_values)
            # Pause to update the plot smoothly (adjust the duration as needed)
            plt.pause(0.001)

            # apply the delta W to W
            for w in range(len(self.weights)):
                self.weights[w] += dynamic_learning_rate * self.weights_delta[w]
                self.weights_delta[w].fill(0)

        plt.show()

        # This alone works!
        # plt.plot(iterations[5:], loss_values[5:])  # Empty plot to be updated later
        # plt.xlabel('Iteration')
        # plt.ylabel('Loss')
        # plt.title('Loss over time')
        # plt.show()

    def train_epoch(self, X, y):
        for i in range(len(X)):
            input_vec = X[i]
            target_vec = y[i]
            # print(f'this_input({input_vec.shape}): {input_vec}')
            # print(f'this_exp_output({target_vec.shape}): {target_vec}')

            ### Feed forward ###
            outputs = self.feed_forward_save_cache(input_vec)
            # print(f'outputs({len(outputs)}): {outputs}')

            # trauma propagation
            # trauma_factor = trauma_learning_rate if target_vec == 0 and iteration >= 1 and iteration <= 2 and shoot_odds_one_in(20) else 1.0
            trauma_factor = 1.0

            ### Backpropagation ###

            # simple cartesian loss func
            loss = get_loss_vec(target_vec, outputs[-1])
            delta = loss * sigmoid_derivative(outputs[-1])
            # print(f'delta({delta.shape}): {delta}')

            for j in range(len(self.weights) - 1, -1, -1):
                # print(f'W_j.shape({self.weights[j].shape})')
                # print(f'delta.shape({delta.shape})')
                self.weights_delta[j] += np.dot(outputs[j + 1], delta) * trauma_factor

                interim_loss = np.dot(delta, self.weights[j].T)
                # print(f'adjust({interim_loss.shape}): ...')
                delta = interim_loss * sigmoid_derivative(outputs[j])


# =============================================
rate = 0.000001

mlp = MLP()
mlp.add_layer(input_size=3, output_size=4)
# mlp.add_layer(input_size=4, output_size=5)
# mlp.add_layer(input_size=5, output_size=4)
mlp.add_layer(input_size=4, output_size=3)


def normalize(_d, to_sum=True, copy=True):
    # d is a (n x dimension) np array
    d = _d if not copy else np.copy(_d)
    magnitude = np.linalg.norm(d, axis=1, keepdims=True)
    d /= magnitude
    return d


train_x = (20 * np.random.random((2000, 3)) - 10) / 10
train_y = normalize(train_x)

test_x = 20 * np.random.random((1000, 3)) - 10
test_y = normalize(test_x)

print(train_x.shape)
print(train_y.shape)

print("before", f"loss: {mlp.get_cum_loss(train_y, train_x)}")
mlp.train(train_y, train_y, num_epochs=1000, learning_rate=rate)
print("after", f"loss: {mlp.get_cum_loss(train_y, train_x)}")
# mlp.train(test_x, test_y, epochs=100, learning_rate=rate)
# mlp.train(train_x, train_y, epochs=200, learning_rate=rate)
# mlp.train(test_x, test_y, epochs=10, learning_rate=rate)

# print('reversing...')
# mlp.reverse_weights_if_possible()
#
# mlp.train(train_x, train_y, epochs=1000, learning_rate=rate)
# print('after', f'loss: {mlp.get_cum_loss(test_x, test_y)}')


# =============================================
# from mnist import load
# training_images, training_labels, test_images, test_labels = load()
#
# print(training_images.shape)
# print(training_labels.shape)
#
# rate = 0.000001
#
# mnistMLP = MLP()
# mnistMLP.add_layer(784, 16)
# # mnistMLP.add_layer(16, 16)
# mnistMLP.add_layer(16, 10)
#
# print('before', f'loss: {mnistMLP.get_cum_loss(training_images, training_labels)}')
# mnistMLP.train(training_images, training_labels, num_epochs=1000, learning_rate=rate)
# print('reversing...')
# mnistMLP.reverse_weights_if_possible()
#
# # mnistMLP.train(training_images, training_labels, epochs=100, learning_rate=rate)
# print('after', f'loss: {mnistMLP.get_cum_loss(training_images, training_labels)}')
