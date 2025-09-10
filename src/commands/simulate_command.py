from argparse import ArgumentParser
from argparse import Namespace
from src.commands.command import (
    Command,
    add_hex_only_arguments,
    validate_hex_only_arguments,
    add_training_arguments,
    validate_training_arguments,
    add_global_arguments,
    validate_global_arguments,
    get_dataset,
)
from src.networks.HexagonalNetwork import HexagonalNeuralNetwork
from src.networks.activation.activations import get_activation_function
from src.networks.loss.loss import get_loss_function

import numpy as np


class SimulateCommand(Command):

    def name(self) -> str:
        return "sim"

    def help(self) -> str:
        return "Ad Hoc: Mostly for testing. Simulates a Hexagonal Neural Network being trained and tested and shows figures"

    def configure_parser(self, parser: ArgumentParser):
        add_hex_only_arguments(parser)
        add_training_arguments(parser)
        add_global_arguments(parser)

    def validate_args(self, args: Namespace):
        validate_hex_only_arguments(args)
        validate_training_arguments(args)
        validate_global_arguments(args)

    def invoke(self, args: Namespace):
        loss_function = get_loss_function(args.loss)
        activation_function = get_activation_function(args.activation)
        net = HexagonalNeuralNetwork(
            n=args.n,
            r=args.rotation,
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

        # alternate between rotations 0 and 1
        for i in range(10):
            net.rotate(0)
            # net.graph_weights(activation_only=False, detail=f"alternate_rotation_iteration_{i}")
            net.train_animated(data, epochs=args.epochs, pause=args.pause)
            net.rotate(1)
            net.train_animated(data, epochs=args.epochs, pause=args.pause)
            net.rotate(2)
            net.train_animated(data, epochs=args.epochs, pause=args.pause)

        # net.graph_weights(activation_only=False, detail="untrained")
        # net._graph_multi_activation(r_list=[0, 1])
        # net.train_animated(data, epochs=args.epochs, pause=args.pause)

        # net.graph_weights(activation_only=False, detail="trained")
        # net.graph_weights(activation_only=False, detail="prerotation")
        # net.rotate(1)

        # net.graph_weights(activation_only=False, detail="postrotation")

        # net.train_animated(data, epochs=args.epochs, pause=args.pause)
        # net.graph_weights(activation_only=False, detail="postrotation")
