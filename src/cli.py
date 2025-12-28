import argparse

from commands.reference_command import ReferenceCommand
from commands.adhoc_command import AdhocCommand
from commands.train_command import TrainCommand
from commands.stats_conmand import StatsCommand

def parse_args():
    parser = argparse.ArgumentParser(description="Hexagonal Neural Network CLI tool")

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    commands = [
        ReferenceCommand(),
        AdhocCommand(),
        TrainCommand(),
        StatsCommand()
    ]

    for command in commands:
        subparser = subparsers.add_parser(command.name(), help=command.help())
        subparser.set_defaults(command=command)
        command.configure_parser(subparser)

    args = parser.parse_args()
    
    if "command" not in args or args.command is None:
        parser.print_help()
        print('Command not provided')
        exit(1)
    
    command = args.command

    return args, command

def main():
    args, command = parse_args()
    command(args)

if __name__ == "__main__":
    main()