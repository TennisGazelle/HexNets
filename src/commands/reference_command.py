from argparse import ArgumentParser
from argparse import Namespace
from src.commands.command import Command, add_n_argument, validate_n_argument
from src.networks.HexagonalNetwork import HexagonalNeuralNetwork

class ReferenceCommand(Command):

    def name(self) -> str:
        return "ref"

    def help(self) -> str:
        return "Run a reference implementation of the Hexagonal Neural Network"
    
    def configure_parser(self, parser: ArgumentParser):
        add_n_argument(parser)

        parser.add_argument(
            "-g", 
            "--graph",
            help="which type of graph to output",
            choices=["dot", "matplotlib", "activation", "weight", "terminal_indices"],
            default="matplotlib",
            dest="graph"
        )
        parser.add_argument(
            "--detail",
            help="Subtitle the graph with a detail string",
            default="",
            dest="detail"
        )
    
    def validate_args(self, args: Namespace):
        validate_n_argument(args)
    
    def invoke(self, args: Namespace):
        net = HexagonalNeuralNetwork(n=args.n)

        if args.graph == "dot":
            print("This assumes you have Graphviz installed...")
            output_file = net.graph_dot()
            print(f"Graph saved to {output_file}")
            print(f"Note: Dot file outputted to {output_file.replace('.png', '.dot')}")

        elif args.graph == "matplotlib":
            output_file = net.graphHex()
            print(f"Graph saved to {output_file}")

        elif args.graph == "activation":
            output_file = net._graphW(activation_only=True, detail=args.detail)
            print(f"Graph saved to {output_file}")

        elif args.graph == "weight":
            output_file = net._graphW(activation_only=False, detail=args.detail)
            print(f"Graph saved to {output_file}")

        elif args.graph == "terminal_indices":
            net._printIndices()

        else:
            raise ValueError(f"Invalid graph type: {args.graph}")
        