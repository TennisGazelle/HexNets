from abc import ABC, abstractmethod
from argparse import ArgumentParser, Namespace
import random
import numpy as np

from networks.activation.activations import get_available_activation_functions
from networks.loss.loss import get_available_loss_functions
from data.dataset import IdentityDataset

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


identity_dataset = IdentityDataset()

def get_dataset(n, train_samples, type="identity", scale=1.0):
    if type == "identity":
        return identity_dataset.get_data()
        X = (np.random.rand(train_samples, n) * 2 - 1).astype(float)
        Y = X.copy()
    elif type == "linear":
        X = (np.random.rand(train_samples, n) * 2 - 1).astype(float)
        Y = X.copy() * scale
    else:
        raise ValueError(f"Invalid dataset type: {type}")
    return list(zip(X, Y))


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


def add_hex_only_arguments(parser: ArgumentParser):
    parser.add_argument(
        "-n",
        "--num_dims",
        type=int,
        default=3,
        help="Number of input and output nodes",
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

def add_global_arguments(parser: ArgumentParser):
    parser.add_argument(
        "-m",
        "--model",
        help="Model to use",
        type=str,
        default="hex",
        choices=["hex", "mlp"],
        dest="model",
    )

    parser.add_argument(
        "-s",
        "--seed",
        help="Seed for the random number generator",
        type=int,
        default=42,
        dest="seed",
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


def add_training_arguments(parser: ArgumentParser):
    parser.add_argument(
        "-lr",
        "--learning-rate",
        help="Initial learning rate for the network",
        type=float,
        default=0.01,
        dest="learning_rate",
    )

    parser.add_argument(
        "-e",
        "--epochs",
        help="Number of epochs to train for",
        type=int,
        default=100,
        dest="epochs",
    )

    parser.add_argument(
        "-p",
        "--pause",
        help="Pause between epochs",
        type=float,
        default=0.05,
        dest="pause",
    )

    parser.add_argument(
        "-t",
        "--type",
        help="Type of dataset to use",
        choices=["identity", "linear"],
        default="identity",
        dest="type",
    )

    parser.add_argument(
        "-ds",
        "--dataset-size",
        help="Number of samples in the dataset",
        type=int,
        default=250,
        dest="dataset_size",
    )

    parser.add_argument(
        "--dry-run",
        help="What would be run, do not create a run.",
        default=False,
        action="store_true",
        dest="dry_run"
    )


def validate_hex_only_arguments(args: Namespace):
    if args.n < 2:
        raise ValueError("Number of input nodes must be at least 2")
    if args.rotation < 0 or args.rotation > 5:
        raise ValueError(f"Invalid rotation input: {args.rotation}. Must be a value between 0 and 5.")


def validate_global_arguments(args: Namespace):
    if args.activation not in get_available_activation_functions():
        raise ValueError(
            f"Invalid activation function: {args.activation}. Must be one of: {get_available_activation_functions()}"
        )
    if args.loss not in get_available_loss_functions():
        raise ValueError(f"Invalid loss function: {args.loss}. Must be one of: {get_available_loss_functions()}")


def validate_training_arguments(args: Namespace):
    # random number generator seed
    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.epochs < 1:
        raise ValueError("Number of epochs must be at least 1")
    if args.pause < 0:
        raise ValueError("Pause must be at least 0")
    if args.learning_rate <= 0:
        raise ValueError("Learning rate must be greater than 0")
    if args.dataset_size < 10:
        raise ValueError("Dataset size must be at least 10")
    if args.type not in ["identity", "linear"]:
        raise ValueError(f"Invalid dataset type: {args.type}. Must be one of: identity, linear")
