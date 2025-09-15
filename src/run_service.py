from argparse import Namespace
from pathlib import Path
from datetime import date, datetime
import hashlib
import pathlib
import json
import uuid
from typing import Tuple, Union

from networks.activation.activations import get_activation_function
from networks.loss.loss import get_loss_function
from networks.MLPNetwork import MLPNetwork
from networks.HexagonalNetwork import HexagonalNeuralNetwork


class RunService:
    def __init__(self, args):
        if "run_dir" not in args or args.run_dir is None or (args.run_dir and not args.run_dir.exists()):
            timestamp, run_folder_name = RunService.make_run_folder_name(args.run_dir if args.run_dir else None)
            self.run_folder_path = pathlib.Path(f"runs/{run_folder_name}")
            self.config_path = self.run_folder_path / "config.json"
            self.manifest_path = self.run_folder_path / "manifest.json"
            self.training_metrics_path = self.run_folder_path / "training_metrics.json"

            self.loss_function = get_loss_function(args.loss)
            self.activation_function = get_activation_function(args.activation)

            self.config_contents = {
                "model_type": args.model,
                "activation_type": self.activation_function.display_name,
                "loss_type": self.loss_function.display_name,
                "learning_rate": args.learning_rate,
                "epochs": args.epochs,
                "dataset_type": args.type,
                "dataset_size": args.dataset_size,
                "run_folder_name": run_folder_name,
                "model_metadata": {},
            }

            self.manifest_contents = {
                "model_hash": RunService.get_model_hash(args),
                "data_hash": RunService.get_data_hash(args),
                "date_first_run": timestamp,
            }

            self.training_metrics_contents = None

            if args.model == "mlp":
                self.config_contents["model_metadata"]["hidden_dims"] = [4, 5, 4]
                self.net = MLPNetwork(
                    input_dim=args.n,
                    output_dim=args.n,
                    hidden_dims=[4, 5, 4],
                    learning_rate=args.learning_rate,
                    activation=self.activation_function,
                    loss=self.loss_function,
                )
            elif args.model == "hex":
                self.config_contents["model_metadata"]["n"] = args.n
                self.config_contents["model_metadata"]["r"] = args.rotation
                self.net = HexagonalNeuralNetwork(
                    n=args.n,
                    r=args.rotation,
                    learning_rate=args.learning_rate,
                    activation=self.activation_function,
                    loss=self.loss_function,
                )
            else:
                raise ValueError(f"Invalid model: {args.model}")

        else:
            self.run_folder_path = args.run_dir
            self.config_path = self.run_folder_path / "config.json"
            self.manifest_path = self.run_folder_path / "manifest.json"
            self.training_metrics_path = self.run_folder_path / "training_metrics.json"

            # load the config, manifest, and training metrics
            if not self.config_path.exists():
                raise ValueError(f"Expected config file missing: {self.config_path}")
            else:
                with open(self.config_path, "r") as f:
                    self.config_contents = json.load(f)
                self.config_contents["model_metadata"] = self.config_contents["model_metadata"]

            if not self.manifest_path.exists():
                raise ValueError(f"Expected manifest file missing: {self.manifest_path}")
            else:
                with open(self.manifest_path, "r") as f:
                    self.manifest_contents = json.load(f)

            if not self.training_metrics_path.exists():
                raise ValueError(f"Expected training file missing: {self.training_metrics_path}")
            else:
                with open(self.training_metrics_path, "r") as f:
                    self.training_metrics_contents = json.load(f)

            self.loss_function = get_loss_function(self.config_contents["loss_type"])
            self.activation_function = get_activation_function(self.config_contents["activation_type"])

            # load the network
            if self.config_contents["model_type"] == "mlp":
                self.net = MLPNetwork(
                    input_dim=self.config_contents["model_metadata"]["input_dim"],
                    output_dim=self.config_contents["model_metadata"]["output_dim"],
                    hidden_dims=self.config_contents["model_metadata"]["hidden_dims"],
                    learning_rate=self.config_contents["learning_rate"],
                    activation=self.activation_function,
                    loss=self.loss_function,
                )
                self.net.load(self.get_network_weights_path())
            elif self.config_contents["model_type"] == "hex":
                self.net = HexagonalNeuralNetwork(
                    n=self.config_contents["model_metadata"]["n"],
                    r=self.config_contents["model_metadata"]["r"],
                    learning_rate=self.config_contents["learning_rate"],
                    activation=self.activation_function,
                    loss=self.loss_function,
                )
                self.net.load(self.get_network_weights_path())

    def set_training_metrics(self, training_metrics: dict):
        self.training_metrics_contents = training_metrics.copy()

    def get_network_weights_path(self) -> pathlib.Path:
        return self.run_folder_path / "model.pkl"

    def get_figures_path(self) -> pathlib.Path:
        return self.run_folder_path / "plots"

    def print_paths(self) -> None:
        print(self.run_folder_path)
        print("-\t", self.config_path)
        print("-\t", self.manifest_path)
        print("-\t", self.training_metrics_path)

    def print_last_training_metrics(self) -> None:
        print(f"After {self.config_contents.get('epochs')} epochs:")
        print(f"Loss ({self.config_contents.get('loss_type')}): {self.training_metrics_contents.get('loss')[-1]}")
        print(f"Accuracy (RMSE): {self.training_metrics_contents.get('accuracy')[-1]}")
        print(f"R_2: {self.training_metrics_contents.get('r_squared')[-1]}")

    @staticmethod
    def make_run_folder_name(filename: Union[str, None] = None) -> Tuple[str, str]:
        now = datetime.now().strftime("%Y-%m-%d_%H-%M")
        return now, now + "_" + str(uuid.uuid4())[0:6] if filename is None else filename

    @staticmethod
    def get_model_hash(args: Namespace) -> str:
        if args.model == "hex":
            params = f"{args.n}_{args.rotation}"
        elif args.model == "mlp":
            params = f""
        elif args.model == "conv":
            params = f"{args.conv_filters}_{args.conv_kernel_size}_{args.conv_stride}_{args.conv_padding}"
        elif args.model == "capsule":
            params = f"{args.capsule_num_capsules}_{args.capsule_num_routing_iterations}_{args.capsule_num_outputs}"
        else:
            raise ValueError(f"Invalid model: {args.model}")
        return hashlib.sha256(f"{args.model}_{params}_{args.activation}_{args.loss}".encode()).hexdigest()

    @staticmethod
    def get_data_hash(args: Namespace) -> str:
        return hashlib.sha256(f"{args.type}_{args.dataset_size}".encode()).hexdigest()

    def output_run_files(self) -> None:
        print(f"run data saved to {self.run_folder_path}")

        self.run_folder_path.mkdir(parents=True, exist_ok=True)
        self.get_figures_path().mkdir(parents=True, exist_ok=True)

        with open(self.config_path, "w") as f:
            json.dump(self.config_contents, f, indent=3)

        with open(self.manifest_path, "w") as f:
            json.dump(self.manifest_contents, f, indent=3)

        with open(self.training_metrics_path, "w") as f:
            json.dump(self.training_metrics_contents, f, indent=3)
