import argparse
import logging

from commands.reference_command import ReferenceCommand
from commands.adhoc_command import AdhocCommand
from commands.train_command import TrainCommand
from commands.stats_command import StatsCommand
from services.logging_config import setup_logging, get_logger

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Hexagonal Neural Network CLI tool")

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    commands = [ReferenceCommand(), AdhocCommand(), TrainCommand(), StatsCommand()]

    for command in commands:
        subparser = subparsers.add_parser(command.name(), help=command.help())
        subparser.set_defaults(command=command)
        command.configure_parser(subparser)

    args = parser.parse_args()

    if "command" not in args or args.command is None:
        parser.print_help()
        logger.error("Command not provided")
        exit(1)

    command = args.command

    return args, command


def main():
    # Initialize logging
    setup_logging(level=logging.DEBUG)
    args, command = parse_args()
    try:
        command(args)
    except ValueError as e:
        logger.error("%s", e)
        raise SystemExit(1) from e


if __name__ == "__main__":
    main()
