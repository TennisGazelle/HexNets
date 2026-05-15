from argparse import ArgumentParser
from argparse import Namespace
import logging
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
from services.run_service import RunService
from services.run_config import RunConfig
from services.logging_config import get_logger

logger = get_logger(__name__)

# Default ``scale`` for ``linear_scale`` dataset (CLI may add ``--dataset-scale`` later).
LINEAR_SCALE_DEFAULT = 2.0


class TrainCommand(Command):
    def name(self):
        return "train"

    def help(self):
        return "Train the network and save the model to a file"

    def configure_parser(self, parser: ArgumentParser):
        add_hex_only_arguments(parser)
        add_training_arguments(parser)
        add_global_arguments(parser)
        # parser.add_argument("data", type=str, help="The data to train the network on", choices=["identity", "linear_scale"])

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
            "--run-name",
            type=str,
            default=None,
            required=False,
            help="name of the new run",
            dest="run_name",
        )

        parser.add_argument(
            "--run-note",
            type=str,
            default=None,
            required=False,
            help="Optional freeform note stored in the run manifest (paper traceability).",
            dest="run_note",
        )

        parser.add_argument(
            "--run-tags",
            type=str,
            default=None,
            required=False,
            help='Optional comma-separated tags stored in manifest (e.g. "paper,v1").',
            dest="run_tags",
        )

        parser.add_argument(
            "-rc",
            "--run-config",
            type=pathlib.Path,
            default=None,
            required=False,
            help="Start a new run from this config.json (same schema as runs/.../config.json). CLI flags override the file.",
            dest="run_config",
        )

        parser.add_argument(
            "--run-config-json",
            type=str,
            default=None,
            required=False,
            help="Same as --run-config but pass a JSON object as a string (shell-escape carefully). CLI flags override.",
            dest="run_config_json",
        )

        parser.add_argument(
            "-f",
            "--force",
            default=False,
            action="store_true",
            help="Force overwrite of existing run",
            dest="force_overwrite",
        )

    def __call__(self, args: Namespace):
        RunConfig.validate_cli_sources(args)
        if getattr(args, "run_config", None) is not None or getattr(args, "run_config_json", None) is not None:
            template = RunConfig.from_cli_sources(args)
            args = template.merged_train_namespace(original=args)
        self.validate_args(args)
        self.invoke(args)

    def validate_args(self, args: Namespace):
        validate_hex_only_arguments(args)
        validate_training_arguments(args)
        validate_global_arguments(args)

        if args.run_name:
            if args.run_dir:
                raise ValueError("Cannot define desired run-name and have a run_dir, pick one.")

            if (RunService.runs_dir / args.run_name).exists():
                if getattr(args, "force_overwrite", False):
                    logger.warning(f"Run named '{args.run_name}' already exists, forcing overwrite")
                else:
                    raise ValueError(f"Run named '{args.run_name}' already exists")

        if args.run_dir:
            if not args.run_dir.exists():
                raise ValueError(f"Run Dir '{args.run_dir}' does not exist, use --run-name if it's meant to be new.")

    def invoke(self, args: Namespace):
        if args.type == "linear_scale":
            args.dataset_scale = LINEAR_SCALE_DEFAULT
            effective_scale = LINEAR_SCALE_DEFAULT
        elif args.type == "diagonal_scale":
            args.dataset_scale = 1.0
            effective_scale = 1.0
        else:
            args.dataset_scale = None
            effective_scale = 1.0

        data = get_dataset(
            args.n,
            args.dataset_size,
            type=args.type,
            scale=effective_scale,
            noise_mode=getattr(args, "dataset_noise", None),
            noise_mu=getattr(args, "dataset_noise_mu", 0.0),
            noise_sigma=getattr(args, "dataset_noise_sigma", 0.1),
            noise_seed=args.seed,
        )

        run = RunService(args)
        net = run.net
        net.show_stats()

        if args.dry_run:
            logger.info("Dry run only, none of the above files were created")
            return

        run.output_run_files()

        train_show = getattr(args, "train_show", "metrics")
        show_training_metrics = train_show in ("metrics", "both")
        show_weights_live = train_show in ("weights", "both")

        net.train_animated(
            data,
            epochs=args.epochs,
            pause=args.pause,
            output_dir=run.get_figures_path(),
            simple_figure_names=True,
            show_training_metrics=show_training_metrics,
            show_weights_live=show_weights_live,
        )

        run.set_training_metrics(net.get_metrics_json())
        net.save(run.get_network_weights_path())

        run.output_run_files()
