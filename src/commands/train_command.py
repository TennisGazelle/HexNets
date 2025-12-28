from argparse import ArgumentParser
from argparse import Namespace
import pathlib
from commands.command import (
    Command,
    validate_hex_only_arguments,
    add_hex_only_arguments,
    add_training_arguments,
    validate_training_arguments,
    add_global_arguments,
    validate_global_arguments,
    get_dataset,
)
from run_service import RunService


class TrainCommand(Command):
    def name(self):
        return "train"

    def help(self):
        return "Train the network and save the model to a file"

    def configure_parser(self, parser: ArgumentParser):
        add_hex_only_arguments(parser)
        add_training_arguments(parser)
        add_global_arguments(parser)
        # parser.add_argument("data", type=str, help="The data to train the network on", choices=["identity", "linear"])

        # parser.add_argument(
        #     "-o",
        #     "--output-name",
        #     type=str,
        #     help="The name of the run to save the model to (if not provided, the model will not be saved)",
        #     default=None,
        #     dest="output_name",
        # )

        # parser.add_argument(
        #     "-i",
        #     "--input-dir",
        #     type=pathlib.Path,
        #     help="The directory to load the model from (if provided, epoch number suffix will be updated)",
        #     default=None,
        #     dest="input_dir",
        # )

        parser.add_argument(
            "-rd",
            "--run-dir",
            type=pathlib.Path,
            default=None,
            required=False,
            help="The run to load the model from",
            dest="run_dir",
        )

        parser.add_argument(
            "-rn",
            "--run_name",
            type=str,
            default=None,
            required=False,
            help="name of the new run",
            dest="run_name"
        )

    def validate_args(self, args: Namespace):
        validate_hex_only_arguments(args)
        validate_training_arguments(args)
        validate_global_arguments(args)

        if args.run_name:
            if args.run_dir:
                raise ValueError("Cannot define desired run_name and have a run_dir, pick one.")

            if (RunService.runs_dir / args.run_name).exists():
                raise ValueError(f"Run named '{args.run_name}' already exists")

        if args.run_dir:
            if not args.run_dir.exists():
                raise ValueError(f"Run Dir '{args.run_dir}' does not exist, use --run_name if it's meant to be new.")

    def invoke(self, args: Namespace):
        if args.type == "identity":
            data = get_dataset(args.n, args.dataset_size, type="identity")
        elif args.type == "linear":
            data = get_dataset(args.n, args.dataset_size, type="linear", scale=2.0)
        else:
            raise ValueError(f"Invalid dataset type: {args.type}")

        run = RunService(args)
        net = run.net
        run.print_paths()
        net.show_stats()

        if args.dry_run:
            print("Dry run only, none of the above files were created")
            return

        run.output_run_files()

        # net.graph_structure(output_dir=run.get_figures_path())
        net.graph_weights(activation_only=False, output_dir=run.get_figures_path())
        net.train_animated(data, epochs=args.epochs, pause=args.pause, output_dir=run.get_figures_path())
        net.graph_weights(activation_only=False, output_dir=run.get_figures_path(), detail="trained")

        run.set_training_metrics(net.get_metrics_json())
        net.save(run.get_network_weights_path())

        run.output_run_files()
