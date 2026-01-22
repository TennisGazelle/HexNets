from abc import ABC, abstractmethod
from argparse import ArgumentParser, Namespace
import random
import numpy as np
import logging

from networks.activation.activations import get_available_activation_functions
from networks.loss.loss import get_available_loss_functions
from networks.learning_rate.learning_rate import get_available_learning_rates
from data.dataset import IdentityDataset, LinearScaleDataset

logger = logging.getLogger(__name__)

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




def get_dataset(n, train_samples, type="identity", scale=1.0):
    if type == "identity":
        return IdentityDataset(d=n, num_samples=train_samples)
    elif type == "linear":
        return LinearScaleDataset(d=n, num_samples=train_samples, scale=scale)
    else:
        raise ValueError(f"Invalid dataset type: {type}")


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


def add_hex_only_arguments(parser: ArgumentParser, set_defaults: bool = True):
    parser.add_argument(
        "-n",
        "--num_dims",
        type=int,
        default=3 if set_defaults else None,
        help="Number of input and output nodes",
        dest="n",
    )

    parser.add_argument(
        "-r",
        "--rotation",
        help="Value between 0 and 5, (e.g. 0,1,2,3,4,5) of which hexagon rotation to display",
        type=int,
        default=0 if set_defaults else None,
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

    # parser.add_argument(
    #     "-o",
    #     "--optimizer",
    #     help="Optimizer to use",
    #     type=str,
    #     default="constant",
    #     choices=get_available_optimizers(),
    #     dest="optimizer",
    # )


def add_training_arguments(parser: ArgumentParser):
    parser.add_argument(
        "-lr",
        "--learning-rate",
        help="Learning rate function for the network",
        type=str,
        default="constant",
        choices=get_available_learning_rates(),
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

    act = args.activation.lower()
    loss = args.loss.lower()

    # --- soft warnings (behavioral) ---

    # 1) Sigmoid output clamps regression to (0, 1)
    if act == "sigmoid" and loss in {"mse", "logcosh", "huber", "quantile"}:
        logger.warning(
            "Sigmoid output bounds predictions to (0, 1). "
            "If regression targets are not scaled to [0, 1], "
            "expect saturation and biased predictions. "
            "Recommendation: normalize targets or use a linear output."
        )

    # 2) ReLU output forbids negative predictions
    if act == "relu" and loss in {"mse", "logcosh", "huber", "quantile"}:
        logger.warning(
            "ReLU output restricts predictions to [0, ∞). "
            "If regression targets include negative values, "
            "the model cannot represent them. "
            "Recommendation: use a linear or LeakyReLU output."
        )

    # 3) LeakyReLU output introduces asymmetric scaling
    if act == "leakyrelu" and loss in {"mse", "logcosh", "huber", "quantile"}:
        logger.warning(
            "LeakyReLU output applies asymmetric scaling to negative values. "
            "This may bias regression if symmetry is expected. "
            "Recommendation: use a linear output unless asymmetry is intentional."
        )

    # 4) Quantile loss without explicit quantile
    if loss == "quantile" and not hasattr(args, "quantile"):
        logger.warning(
            "Quantile loss selected but no quantile (q) specified. "
            "Default behavior may be unclear. "
            "Recommendation: explicitly set --quantile (e.g., 0.5 for median)."
        )

    # 5) Quantile loss with nonlinear output
    if loss == "quantile" and act in {"sigmoid", "relu"}:
        logger.warning(
            "Quantile loss with constrained output activation may cause "
            "quantile estimates to clamp at output bounds. "
            "Recommendation: use a linear output for unconstrained quantiles."
        )

    # 6) Huber loss without explicit delta
    if loss == "huber" and not hasattr(args, "huber_delta"):
        logger.warning(
            "Huber loss selected without specifying delta. "
            "Loss behavior is scale-dependent. "
            "Recommendation: normalize targets or explicitly set --huber-delta."
        )

def validate_training_arguments(args: Namespace):
    # random number generator seed
    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.epochs < 1:
        raise ValueError("Number of epochs must be at least 1")
    if args.pause < 0:
        raise ValueError("Pause must be at least 0")
    if args.learning_rate not in get_available_learning_rates():
        raise ValueError(f"Invalid learning rate: {args.learning_rate}. Must be one of: {get_available_learning_rates()}")
    if args.dataset_size < 10:
        raise ValueError("Dataset size must be at least 10")
    if args.type not in ["identity", "linear"]:
        raise ValueError(f"Invalid dataset type: {args.type}. Must be one of: identity, linear")
