from argparse import ArgumentParser
from argparse import Namespace
from src.commands.command import Command, add_structure_argument, validate_structure_argument
from src.networks.HexagonalNetwork import HexagonalNeuralNetwork

class ReferenceCommand(Command):

    def name(self) -> str:
        return "ref"

    def help(self) -> str:
        return "Run a reference implementation of the Hexagonal Neural Network"
    
    def configure_parser(self, parser: ArgumentParser):
        add_structure_argument(parser)

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
        validate_structure_argument(args)
    
    def invoke(self, args: Namespace):
        net = HexagonalNeuralNetwork(n=args.n, r=args.rotation, random_init=True)

        if args.graph == "dot":
            print("This assumes you have Graphviz installed...")
            output_file = net._graph_hex_dot()
            print(f"Graph saved to {output_file}")
            print(f"Note: Dot file outputted to {output_file.replace('.png', '.dot')}")

        elif args.graph == "matplotlib":
            output_file = net._graph_hex()
            print(f"Graph saved to {output_file}")

        elif args.graph == "activation":
            output_file = net._graphW(activation_only=True, detail=args.detail)
            print(f"Graph saved to {output_file}")

        elif args.graph == "weight":
            output_file = net._graphW(activation_only=False, detail=args.detail)
            print(f"Graph saved to {output_file}")

        elif args.graph == "terminal_indices":
            # print(f"Layer sizes: {net._hex_layer_sizes(args.n)}")
            # print(f"Layer indices: {net._get_default_layer_indices(args.n)}")
            net._printIndices(args.rotation)
            # print(f"[0] indices: {net._get_layer_indices(args.n, r=0)}")
            # print(f"[1] indices: {net._get_layer_indices(args.n, r=1)}")
            # print(f"[2] indices: {net._get_layer_indices(args.n, r=2)}")
            # print(f"[3] indices: {net._get_layer_indices(args.n, r=3)}")
            # print(f"[4] indices: {net._get_layer_indices(args.n, r=4)}")
            # print(f"[5] indices: {net._get_layer_indices(args.n, r=5)}")
        else:
            raise ValueError(f"Invalid graph type: {args.graph}")
    