from abc import ABC, abstractmethod
from argparse import ArgumentParser, Namespace
import random

from src.networks.activation.activations import get_available_activation_functions
from src.networks.loss.loss import get_available_loss_functions


def print_header():
    header1 = """
                         __     _   
      /\  /\_____  __ /\ \ \___| |_ 
     / /_/ / _ \ \/ //  \/ / _ \ __|
    / __  /  __/>  </ /\  /  __/ |_ 
    \/ /_/ \___/_/\_\_\ \/ \___|\__|
    """

    header2 = """
        __  __          _   __     __ 
       / / / /__  _  __/ | / /__  / /_
      / /_/ / _ \| |/_/  |/ / _ \/ __/
     / __  /  __/>  </ /|  /  __/ /_  
    /_/ /_/\___/_/|_/_/ |_/\___/\__/  
    """
    print(random.choice([header1, header2]))


class Command(ABC):

    def __call__(self, args: Namespace):
        print_header()
        self.validate_args(args)
        self.invoke(args)

    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def help(self) -> str:
        pass

    @abstractmethod
    def configure_parser(self, parser: ArgumentParser):
        pass

    @abstractmethod
    def validate_args(self, args: Namespace):
        pass

    @abstractmethod
    def invoke(self, args: Namespace):
        pass


def add_structure_argument(parser: ArgumentParser):
    parser.add_argument(
        "-n",
        "--nodes",
        type=int,
        default=3,
        help="Number of input nodes",
        dest="n",
    )

    parser.add_argument(
        "-r",
        "--rotation",
        help="Value between 0 and 5, (e.g. 0,1,2,3,4,5) of which hexagon rotation to display",
        type=int,
        default=0,
        dest="rotation",
    )

    parser.add_argument(
        "-a",
        "--activation",
        help="Activation function to use",
        type=str,
        default="sigmoid",
        choices=get_available_activation_functions(),
        dest="activation",
    )

    parser.add_argument(
        "-l",
        "--loss",
        help="Loss function to use",
        type=str,
        default="mean_squared_error",
        choices=get_available_loss_functions(),
        dest="loss",
    )


def validate_structure_argument(args: Namespace):
    if args.n < 2:
        raise ValueError("Number of input nodes must be at least 2")
    if args.rotation < 0 or args.rotation > 5:
        raise ValueError(f"Invalid rotation input: {args.rotation}. Must be a value between 0 and 5.")
    if args.activation not in get_available_activation_functions():
        raise ValueError(
            f"Invalid activation function: {args.activation}. Must be one of: {get_available_activation_functions()}"
        )
    if args.loss not in get_available_loss_functions():
        raise ValueError(f"Invalid loss function: {args.loss}. Must be one of: {get_available_loss_functions()}")
