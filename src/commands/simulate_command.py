from argparse import ArgumentParser
from argparse import Namespace
from src.commands.command import (
    Command,
    add_structure_argument,
    validate_structure_argument,
    add_training_arguments,
    validate_training_arguments,
)
from src.networks.HexagonalNetwork import HexagonalNeuralNetwork
from src.networks.activation.activations import get_activation_function
from src.networks.loss.loss import get_loss_function

import numpy as np


def get_dataset(n, train_samples, type="identity", scale=1.0):
    if type == "identity":
        X = (np.random.rand(train_samples, n) * 2 - 1).astype(float)
        Y = X.copy()
    elif type == "linear":
        X = (np.random.rand(train_samples, n) * 2 - 1).astype(float)
        Y = X.copy() * scale
    else:
        raise ValueError(f"Invalid dataset type: {type}")
    return list(zip(X, Y))


class SimulateCommand(Command):

    def name(self) -> str:
        return "sim"

    def help(self) -> str:
        return "Ad Hoc: Mostly for testing. Simulates a Hexagonal Neural Network being trained and tested and shows figures"

    def configure_parser(self, parser: ArgumentParser):
        add_structure_argument(parser)
        add_training_arguments(parser)
        parser.add_argument(
            "-t",
            "--type",
            help="Type of dataset to use",
            choices=["identity", "linear"],
            default="identity",
            dest="type",
        )

        parser.add_argument(
            "-ds",
            "--dataset-size",
            help="Number of samples in the dataset",
            type=int,
            default=250,
            dest="dataset_size",
        )

    def validate_args(self, args: Namespace):
        validate_structure_argument(args)
        validate_training_arguments(args)
        if args.dataset_size < 1:
            raise ValueError("Dataset size must be at least 1")

    def invoke(self, args: Namespace):
        loss_function = get_loss_function(args.loss)
        activation_function = get_activation_function(args.activation)
        net = HexagonalNeuralNetwork(
            n=args.n,
            r=args.rotation,
            random_init=True,
            learning_rate=args.learning_rate,
            activation=activation_function,
            loss=loss_function,
        )

        if args.type == "identity":
            data = get_dataset(args.n, args.dataset_size, type="identity")
        elif args.type == "linear":
            data = get_dataset(args.n, args.dataset_size, type="linear", scale=2.0)
        else:
            raise ValueError(f"Invalid dataset type: {args.type}")

        # net.graph_weights(activation_only=False, detail="untrained")
        net.train_animated(data, epochs=args.epochs, pause=args.pause)
        # net.graph_weights(activation_only=False, detail="trained")
