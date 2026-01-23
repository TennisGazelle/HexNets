from argparse import ArgumentParser
from argparse import Namespace
import logging
from pathlib import Path
import numpy as np
from networks.MLPNetwork import MLPNetwork
from src.commands.command import (
    Command,
    add_hex_only_arguments,
    validate_hex_only_arguments,
    add_global_arguments,
    validate_global_arguments,
)
from src.networks.HexagonalNetwork import HexagonalNeuralNetwork
from src.networks.activation.activations import get_activation_function
from src.networks.loss.loss import get_loss_function
from src.networks.learning_rate.learning_rate import get_learning_rate, get_available_learning_rates
from src.figure_service import FigureService
from src.logging_config import get_logger, setup_logging
from src.utils import Colors

logger = get_logger(__name__)


class ReferenceCommand(Command):

    def name(self) -> str:
        return "ref"

    def help(self) -> str:
        return "Run a reference implementation of the Hexagonal Neural Network"

    def configure_parser(self, parser: ArgumentParser):
        add_hex_only_arguments(parser, set_defaults=False)
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
            default=None,
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

        parser.add_argument(
            "--dry-run",
            help="Show what would be generated without actually creating figures.",
            action="store_true",
            dest="dry_run",
        )

    def validate_args(self, args: Namespace):
        # Model 'none' doesn't need validation of n, r, or graph
        if args.model == "none":
            return

        if args.generate_all:
            if args.model == "mlp":
                raise ValueError("--all flag cannot be used with mlp model. Use hex model for batch generation.")
            # Warn if other arguments are provided that will be ignored
            logger.warning(
                f"Warning: --all flag is set. Arguments -n, -r, and -g (if provided) will be ignored. "
                "All graphs will be generated for n=2..8 and r=0..5."
            )
        else:
            # Check if at least something is specified
            # All three (n, r, graph) can be None if not specified
            if args.n is None and args.rotation is None and args.graph is None:
                raise ValueError(
                    "No parameters specified. Please specify at least one of: -n/--num_dims, -r/--rotation, "
                    "or -g/--graph. Use --all to generate all reference graphs."
                )
            # Validate the specified values
            if args.n is not None or args.rotation is not None:
                # Create a temporary args object with defaults for validation
                temp_args = Namespace(**vars(args))
                if temp_args.n is None:
                    temp_args.n = 3  # Use default for validation
                if temp_args.rotation is None:
                    temp_args.rotation = 0  # Use default for validation
                validate_hex_only_arguments(temp_args)
            validate_global_arguments(args)

    def _determine_iteration_ranges(self, args: Namespace):
        """Determine which variables to iterate over based on what's specified.

        Subtractive approach: if a variable is specified, fix it and iterate the others.
        All three parameters (n, r, graph) can be None if not specified.

        Returns: (n_range, r_range, graph_types)
        """
        all_graph_types = [
            "structure_dot",
            "structure_matplotlib",
            "activation",
            "weight",
            "multi_activation",
            "layer_indices_terminal",
        ]

        if args.generate_all:
            # Iterate through all possibilities
            return (range(2, 9), range(6), all_graph_types)

        # Fix what's specified, iterate what's not
        # n: if specified, fix to [args.n], otherwise iterate range(2, 9)
        n_range = [args.n] if args.n is not None else range(2, 9)

        # r: if specified, fix to [args.rotation], otherwise iterate range(6)
        r_range = [args.rotation] if args.rotation is not None else range(6)

        # graph: if specified, fix to [args.graph], otherwise iterate all_graph_types
        graph_types = [args.graph] if args.graph is not None else all_graph_types

        return (n_range, r_range, graph_types)

    def _generate_graph(self, net, graph_type: str, args: Namespace, figures_dir: Path):
        """Generate a single graph of the specified type."""
        if args.dry_run:
            # In dry-run mode, just log what would be generated
            if graph_type == "layer_indices_terminal":
                logger.info(f"[DRY-RUN] Would print layer indices for r={args.rotation}")
            else:
                logger.info(f"[DRY-RUN] Would generate {graph_type} graph for n={net.n}, r={net.r}")
            return

        if graph_type == "structure_dot":
            logger.info("This assumes you have Graphviz installed...")
            output_file, _ = net.graph_structure(output_dir=figures_dir, medium="dot")
            logger.info(f"Graph saved to {output_file}")
            logger.info(f"Note: Dot file outputted to {output_file.replace('.png', '.dot')}")
        elif graph_type == "structure_matplotlib":
            output_file, _ = net.graph_structure(output_dir=figures_dir, medium="matplotlib")
            logger.info(f"Graph saved to {output_file}")
        elif graph_type == "activation":
            output_file, _ = net.graph_weights(activation_only=True, detail=args.detail, output_dir=figures_dir)
            logger.info(f"Graph saved to {output_file}")
        elif graph_type == "weight":
            output_file, _ = net.graph_weights(activation_only=False, detail=args.detail, output_dir=figures_dir)
            logger.info(f"Graph saved to {output_file}")
        elif graph_type == "multi_activation":
            output_file, _ = net._graph_multi_activation(detail=args.detail, output_dir=figures_dir)
            logger.info(f"Graph saved to {output_file}")
        elif graph_type == "layer_indices_terminal":
            net._print_indices(args.rotation)
        else:
            raise ValueError(f"Invalid graph type: {graph_type}")

    def invoke(self, args: Namespace):
        figures_dir = Path("reference")
        figures_dir.mkdir(parents=True, exist_ok=True)

        if args.model == "none":
            # Generate learning rate reference figures
            learning_rates = get_available_learning_rates()
            max_iterations = 500

            print(f"{Colors.BLUE}{'=' * 40}{Colors.NC}")
            print(f"{Colors.BLUE}Generating Learning Rate Reference Figures{Colors.NC}")
            print(f"{Colors.BLUE}  Iterations: {max_iterations}{Colors.NC}")
            print(f"{Colors.BLUE}{'=' * 40}{Colors.NC}")
            print()

            figure_service = FigureService()
            figure_service.set_figures_path(figures_dir)

            total_generated = 0
            for lr_name in learning_rates:
                try:
                    if args.dry_run:
                        print(f"{Colors.YELLOW}  [DRY-RUN] Would generate: {lr_name}...{Colors.NC}")
                        total_generated += 1
                        continue

                    print(f"{Colors.YELLOW}  Generating: {lr_name}...{Colors.NC}")

                    # Create learning rate instance
                    lr_instance = get_learning_rate(lr_name, learning_rate=0.01)

                    # Generate iterations and learning rate values
                    iterations = np.arange(1, max_iterations + 1)
                    lr_values = np.array([lr_instance.rate_at_iteration(i) for i in iterations])

                    # Create figure
                    filename = f"lr_{lr_name}_i{max_iterations}.png"
                    title = f"Learning Rate: {lr_name}"
                    figure = figure_service.init_learning_rate_ref_figure(filename, title, lr_name, max_iterations)

                    # Update figure with data
                    figure.update_figure(iterations, lr_values)

                    # Save figure
                    figure.save_figure()
                    total_generated += 1
                    print(f"{Colors.GREEN}    Saved: {filename}{Colors.NC}")
                except Exception as e:
                    print(f"{Colors.RED}    Error: {e}{Colors.NC}")
                    logger.exception(f"Error generating learning rate figure for {lr_name}")

            print()
            print(f"{Colors.BLUE}{'=' * 40}{Colors.NC}")
            print(f"{Colors.GREEN}Generated {total_generated} learning rate reference figure(s)!{Colors.NC}")
            print(f"{Colors.BLUE}{'=' * 40}{Colors.NC}")
            print()
            print(f"Files saved to: {figures_dir}")
            print()
            return

        if args.model == "mlp":
            # MLP model doesn't support iteration, just generate the single graph
            if args.graph is None:
                raise ValueError("MLP model requires -g/--graph to be specified.")
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
            return

        # Hex model: determine iteration ranges
        n_range, r_range, graph_types = self._determine_iteration_ranges(args)

        activation_function = get_activation_function(args.activation)
        loss_function = get_loss_function(args.loss)

        # Print header if iterating
        is_iterating = len(n_range) > 1 or len(r_range) > 1 or len(graph_types) > 1
        if is_iterating:
            print(f"{Colors.BLUE}{'=' * 40}{Colors.NC}")
            print(f"{Colors.BLUE}Generating Reference Graphs{Colors.NC}")
            if args.generate_all:
                print(f"{Colors.BLUE}  Mode: ALL (n=2..8, r=0..5, all graph types){Colors.NC}")
            else:
                print(f"{Colors.BLUE}  Mode: Iterating over unspecified parameters{Colors.NC}")
            print(f"{Colors.BLUE}{'=' * 40}{Colors.NC}")
            print()

        total_generated = 0

        # Iterate through the determined ranges
        for n in n_range:
            if is_iterating and len(n_range) > 1:
                print(f"{Colors.GREEN}{'=' * 40}{Colors.NC}")
                print(f"{Colors.GREEN}n={n}{Colors.NC}")
                print(f"{Colors.GREEN}{'=' * 40}{Colors.NC}")

            for r in r_range:
                # Create network for this n, r combination
                net = HexagonalNeuralNetwork(
                    n=n,
                    r=r,
                    learning_rate="constant",
                    activation=activation_function,
                    loss=loss_function,
                )

                # Update args.rotation for this iteration (needed for layer_indices_terminal)
                current_rotation = args.rotation
                args.rotation = r

                for graph_type in graph_types:
                    try:
                        if is_iterating:
                            if len(r_range) > 1 or len(graph_types) > 1:
                                print(f"{Colors.YELLOW}  Generating: r={r}, graph={graph_type}...{Colors.NC}")

                        self._generate_graph(net, graph_type, args, figures_dir)
                        total_generated += 1
                    except Exception as e:
                        print(f"{Colors.RED}    Error: {e}{Colors.NC}")
                        logger.exception(f"Error generating graph for n={n}, r={r}, graph={graph_type}")

                # Restore original rotation
                args.rotation = current_rotation

            if is_iterating and len(n_range) > 1:
                print()

        if is_iterating:
            print()
            print(f"{Colors.BLUE}{'=' * 40}{Colors.NC}")
            print(f"{Colors.GREEN}Generated {total_generated} reference graph(s)!{Colors.NC}")
            print(f"{Colors.BLUE}{'=' * 40}{Colors.NC}")
            print()
            print(f"Files saved to: {figures_dir}")
            print()
