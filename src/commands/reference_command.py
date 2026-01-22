from argparse import ArgumentParser
from argparse import Namespace
import logging
from pathlib import Path
from networks.MLPNetwork import MLPNetwork
from commands.command import (
    Command,
    add_hex_only_arguments,
    validate_hex_only_arguments,
    add_global_arguments,
    validate_global_arguments,
)
from networks.HexagonalNetwork import HexagonalNeuralNetwork
from networks.activation.activations import get_activation_function
from networks.loss.loss import get_loss_function
from logging_config import get_logger, setup_logging
from utils import Colors

logger = get_logger(__name__)


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

        parser.add_argument(
            "--all",
            help="Generate all reference graphs for n=2..8 and r=0..5. Ignores -n and -r arguments.",
            action="store_true",
            dest="generate_all",
        )

    def validate_args(self, args: Namespace):
        if args.generate_all:
            # Warn if other arguments are provided that will be ignored
            # Check if user explicitly provided -n or -r (they have defaults, so we check if they differ)
            # We'll warn if they're not the defaults, but this is approximate
            if hasattr(args, '_explicit_n') or hasattr(args, '_explicit_r'):
                # If we had a way to track explicit args, use it
                pass
            # Always warn when --all is used with any explicit n or r
            logger.warning(
                f"Warning: --all flag is set. Arguments -n and -r (if provided) will be ignored. "
                "All graphs will be generated for n=2..8 and r=0..5."
            )
        else:
            validate_hex_only_arguments(args)
            validate_global_arguments(args)

    def _generate_all_refs(self, figures_dir: Path):
        """Generate all reference graphs for n=2..8 and r=0..5."""
        activation_function = get_activation_function("sigmoid")
        loss_function = get_loss_function("mean_squared_error")
        
        print(f"{Colors.BLUE}{'=' * 40}{Colors.NC}")
        print(f"{Colors.BLUE}Generating All Reference Graphs{Colors.NC}")
        print(f"{Colors.BLUE}{'=' * 40}{Colors.NC}")
        print()
        
        # Generate for all n values
        for n in range(2, 9):  # n=2 to n=8
            try:
                self._generate_for_n(n, figures_dir, activation_function, loss_function)
            except Exception as e:
                print(f"{Colors.RED}Error generating graphs for n={n}: {e}{Colors.NC}")
                logger.exception(f"Error generating graphs for n={n}")
                continue
        
        print(f"{Colors.BLUE}{'=' * 40}{Colors.NC}")
        print(f"{Colors.GREEN}All reference graphs generated!{Colors.NC}")
        print(f"{Colors.BLUE}{'=' * 40}{Colors.NC}")
        print()
        print(f"Files saved to: {figures_dir}")
        print()
        print("Summary:")
        print("  - Structure graphs: 6 rotations × 7 n values = 42 files")
        print("  - Activation matrices: 6 rotations × 7 n values = 42 files")
        print("  - Weight matrices: 6 rotations × 7 n values = 42 files")
        print("  - Multi-activation overlays: 7 n values = 7 files")
        print("  - Total: 133 reference files")
        print()

    def _generate_for_n(self, n: int, figures_dir: Path, activation_function, loss_function):
        """Generate all reference graphs for a specific n value."""
        print(f"{Colors.GREEN}{'=' * 40}{Colors.NC}")
        print(f"{Colors.GREEN}n={n}{Colors.NC}")
        print(f"{Colors.GREEN}{'=' * 40}{Colors.NC}")
        
        # Memoize networks by rotation to avoid recreating them
        nets = {}
        for r in range(6):
            nets[r] = HexagonalNeuralNetwork(
                n=n,
                r=r,
                learning_rate="constant",
                activation=activation_function,
                loss=loss_function,
            )
        
        # Layer indices (terminal output)
        print(f"{Colors.YELLOW}  Generating layer indices (terminal output)...{Colors.NC}")
        # Temporarily set logging to DEBUG for layer indices
        logging.getLogger().setLevel(logging.DEBUG)
        for r in range(6):
            print(f"    Rotation {r}:")
            nets[r]._print_indices(r)
        # Set back to WARNING to reduce noise for graph generation
        logging.getLogger().setLevel(logging.WARNING)
        
        # Multi-activation overlay (one per n, r doesn't matter)
        print(f"{Colors.YELLOW}  Generating multi-activation overlay...{Colors.NC}")
        nets[0]._graph_multi_activation(detail="", output_dir=figures_dir)
        
        # Structure graphs (one per rotation)
        print(f"{Colors.YELLOW}  Generating structure graphs...{Colors.NC}")
        for r in range(6):
            print(f"    Rotation {r}...")
            try:
                nets[r].graph_structure(output_dir=figures_dir, medium="matplotlib")
            except Exception as e:
                print(f"{Colors.RED}      Error generating structure graph for r={r}: {e}{Colors.NC}")
                logger.exception(f"Error generating structure graph for n={n}, r={r}")
        
        # Activation matrices (one per rotation)
        print(f"{Colors.YELLOW}  Generating activation matrices...{Colors.NC}")
        for r in range(6):
            print(f"    Rotation {r}...")
            try:
                nets[r].graph_weights(activation_only=True, detail="", output_dir=figures_dir)
            except Exception as e:
                print(f"{Colors.RED}      Error generating activation matrix for r={r}: {e}{Colors.NC}")
                logger.exception(f"Error generating activation matrix for n={n}, r={r}")
        
        # Weight matrices (one per rotation)
        print(f"{Colors.YELLOW}  Generating weight matrices...{Colors.NC}")
        for r in range(6):
            print(f"    Rotation {r}...")
            try:
                nets[r].graph_weights(activation_only=False, detail="", output_dir=figures_dir)
            except Exception as e:
                print(f"{Colors.RED}      Error generating weight matrix for r={r}: {e}{Colors.NC}")
                logger.exception(f"Error generating weight matrix for n={n}, r={r}")
        
        print()

    def invoke(self, args: Namespace):
        if args.generate_all:
            # Generate all reference graphs
            figures_dir = Path("reference")
            figures_dir.mkdir(parents=True, exist_ok=True)
            self._generate_all_refs(figures_dir)
            return

        if args.model == "hex":
            activation_function = get_activation_function(args.activation)
            loss_function = get_loss_function(args.loss)
            net = HexagonalNeuralNetwork(
                n=args.n,
                r=args.rotation,
                learning_rate="constant",
                activation=activation_function,
                loss=loss_function,
            )

            if args.graph == "structure_dot":
                logger.info("This assumes you have Graphviz installed...")
                output_file, _ = net.graph_structure(medium="dot")
                logger.info(f"Graph saved to {output_file}")
                logger.info(f"Note: Dot file outputted to {output_file.replace('.png', '.dot')}")

            elif args.graph == "structure_matplotlib":
                output_file, _ = net.graph_structure(output_dir="reference", medium="matplotlib")
                logger.info(f"Graph saved to {output_file}")

            elif args.graph == "activation":
                output_file, _ = net.graph_weights(activation_only=True, detail=args.detail, output_dir="reference")
                logger.info(f"Graph saved to {output_file}")

            elif args.graph == "weight":
                output_file, _ = net.graph_weights(activation_only=False, detail=args.detail, output_dir="reference")
                logger.info(f"Graph saved to {output_file}")

            elif args.graph == "multi_activation":
                output_file, _ = net._graph_multi_activation(detail=args.detail, output_dir="reference")
                logger.info(f"Graph saved to {output_file}")

            elif args.graph == "layer_indices_terminal":
                net._print_indices(args.rotation)
            else:
                raise ValueError(f"Invalid graph type: {args.graph}")
        elif args.model == "mlp":
            activation_function = get_activation_function(args.activation)
            loss_function = get_loss_function(args.loss)
            net = MLPNetwork(
                input_dim=3,
                output_dim=3,
                hidden_dims=[10, 10],
                learning_rate="constant",
                activation=activation_function,
                loss=loss_function,
            )
            if args.graph == "structure_matplotlib":
                output_file = net.graph_structure()
                logger.info(f"Graph saved to {output_file}")
            else:
                logger.warning(f"Not Yet Implemented: {args.graph}")
