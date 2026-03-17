import os
import pathlib
import shutil
from argparse import ArgumentParser
from argparse import Namespace

from commands.command import Command

from services.run_service import RunService


class StatsCommand(Command):
    def name(self):
        return "stats"

    def help(self):
        return "Load and Parse and Run and print statistics of the model or the run itself"

    def configure_parser(self, parser: ArgumentParser):
        parser.add_argument(
            "run_dir",
            type=pathlib.Path,
            help="The run to get stats from",
        )

    def validate_args(self, args: Namespace):
        if not args.run_dir.exists():
            raise ValueError(f"Given Run Dir does not exist: {args.run_dir}")

    def invoke(self, args: Namespace):
        run = RunService(args)
        run.net.show_stats()
        run.print_last_training_metrics()
