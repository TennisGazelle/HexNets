from argparse import ArgumentParser
from argparse import Namespace
from src.networks.loss.loss import get_loss_function
from src.networks.activation.activations import get_activation_function
from src.commands.command import (
    Command,
    validate_structure_argument,
    add_structure_argument,
    add_training_arguments,
    validate_training_arguments,
)
from src.networks.HexagonalNetwork import HexagonalNeuralNetwork
from src.networks.MLPNetwork import MLPNetwork
from src.commands.simulate_command import get_dataset


class TrainCommand(Command):
    def name(self):
        return "train"

    def help(self):
        return "Train the network and save the model to a file"

    def configure_parser(self, parser: ArgumentParser):
        add_structure_argument(parser)
        add_training_arguments(parser)
        parser.add_argument("network", type=str, help="The network to train", choices=["hexagonal", "mlp"])
        parser.add_argument("data", type=str, help="The data to train the network on")

    def validate_args(self, args: Namespace):
        validate_structure_argument(args)
        validate_training_arguments(args)

        if args.network not in ["hex", "mlp"]:
            raise ValueError(f"Invalid network: {args.network}")

        if args.data not in ["identity", "linear"]:
            raise ValueError(f"Invalid data: {args.data}")

    def invoke(self, args: Namespace):
        loss_function = get_loss_function(args.loss)
        activation_function = get_activation_function(args.activation)

        data = get_dataset(3, 250, type="identity")

        net = MLPNetwork(
            input_dim=3,
            output_dim=3,
            hidden_dims=[4, 5, 4],
            learning_rate=args.learning_rate,
            activation=activation_function,
            loss=loss_function,
        )
        net.train_animated(data, epochs=args.epochs, pause=args.pause)

        # if args.network == "hex":
        #     net = HexagonalNeuralNetwork(n=args.n, r=args.rotation, learning_rate=args.learning_rate, activation=args.activation, loss=args.loss)
        # elif args.network == "mlp":
        #     net = MLPNetwork(learning_rate=args.learning_rate, activation=args.activation, loss=args.loss)
        # else:
        #     raise ValueError(f"Invalid network: {args.network}")

        # todo: make data part of training arguments
        # make animation optional in this command only
        #
        # net.train_animated(data, epochs=args.epochs, pause=args.pause)
