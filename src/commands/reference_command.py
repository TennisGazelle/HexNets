from argparse import ArgumentParser
from argparse import Namespace
from src.networks.MLPNetwork import MLPNetwork
from src.commands.command import (
    Command,
    add_hex_only_arguments,
    validate_hex_only_arguments,
    add_global_arguments,
    validate_global_arguments,
)
from src.networks.HexagonalNetwork import HexagonalNeuralNetwork


class ReferenceCommand(Command):

    def name(self) -> str:
        return "ref"

    def help(self) -> str:
        return "Run a reference implementation of the Hexagonal Neural Network"

    def configure_parser(self, parser: ArgumentParser):
        add_hex_only_arguments(parser)
        add_global_arguments(parser)

        parser.add_argument(
            "-g",
            "--graph",
            help="which type of graph to output",
            choices=[
                "structure_dot",
                "structure_matplotlib",
                "activation",
                "weight",
                "multi_activation",
                "layer_indices_terminal",
            ],
            default="structure_matplotlib",
            dest="graph",
        )

        parser.add_argument(
            "--detail",
            help="Subtitle the graph with a detail string",
            default="",
            dest="detail",
        )

    def validate_args(self, args: Namespace):
        validate_hex_only_arguments(args)
        validate_global_arguments(args)

    def invoke(self, args: Namespace):
        if args.model == "hex":
            net = HexagonalNeuralNetwork(n=args.n, r=args.rotation, activation=args.activation, loss=args.loss)

            if args.graph == "structure_dot":
                print("This assumes you have Graphviz installed...")
                output_file = net.graph_structure(medium="dot")
                print(f"Graph saved to {output_file}")
                print(f"Note: Dot file outputted to {output_file.replace('.png', '.dot')}")

            elif args.graph == "structure_matplotlib":
                output_file = net.graph_structure(medium="matplotlib")
                print(f"Graph saved to {output_file}")

            elif args.graph == "activation":
                output_file = net.graph_weights(activation_only=True, detail=args.detail)
                print(f"Graph saved to {output_file}")

            elif args.graph == "weight":
                output_file = net.graph_weights(activation_only=False, detail=args.detail)
                print(f"Graph saved to {output_file}")

            elif args.graph == "multi_activation":
                output_file = net._graph_multi_activation(detail=args.detail)
                print(f"Graph saved to {output_file}")

            elif args.graph == "layer_indices_terminal":
                net._print_indices(args.rotation)
            else:
                raise ValueError(f"Invalid graph type: {args.graph}")
        elif args.model == "mlp":
            net = MLPNetwork(
                input_dim=3,
                output_dim=3,
                hidden_dims=[10, 10],
                activation=args.activation,
                loss=args.loss,
            )
            if args.graph == "structure_matplotlib":
                output_file = net.graph_structure()
                print(f"Graph saved to {output_file}")
            else:
                print(f"Not Yet Implemented: {args.graph}")
