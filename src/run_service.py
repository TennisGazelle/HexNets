from argparse import Namespace
from pathlib import Path
from datetime import date, datetime
import hashlib
import pathlib
import json
import uuid


class RunService:
    def __init__(self, args):
        if "run_dir" not in args or args.run_dir is None:
            timestamp, run_folder_name = RunService.make_run_folder_name()
            self.run_folder_path = pathlib.Path(f"runs/{run_folder_name}")
            self.config_path = self.run_folder_path / "config.json"
            self.manifest_path = self.run_folder_path / "manifest.json"
            self.training_metrics_path = self.run_folder_path / "training.json"

            self.config_contents = {
                "model_type": args.model,
                "activation_type": args.activation,
                "loss_type": args.loss,
                "learning_rate": args.learning_rate,
                "epochs": args.epochs,
                "dataset_type": args.type,
                "dataset_size": args.dataset_size,
                "run_folder_name": run_folder_name,
            }

            self.manifest_contents = {
                "model_hash": RunService.get_model_hash(args),
                "data_hash": RunService.get_data_hash(args),
                "date_first_run": timestamp,
            }

            self.training_metrics = None

        else:
            self.run_folder_path = args.run_dir
            self.config_path = self.run_folder_path / "config.json"
            self.manifest_path = self.run_folder_path / "manifest.json"
            self.training_metrics_path = self.run_folder_path / "training.json"

            if not self.config_path.exists():
                raise ValueError(f"Expected config file missing: {self.config_path}")
            else:
                with open(self.config_path, "r") as f:
                    self.config_contents = json.load(f)

            if not self.manifest_path.exists():
                raise ValueError(f"Expected manifest file missing: {self.manifest_path}")
            else:
                with open(self.manifest_path, "r") as f:
                    self.manifest_contents = json.load(f)

            if not self.training_metrics_path.exists():
                raise ValueError(f"Expected training file missing: {self.training_metrics_path}")
            else:
                with open(self.training_metrics_path, "r") as f:
                    self.training_metrics = json.load(f)

    def set_training_metrics(self, training_metrics: dict):
        self.training_metrics = training_metrics.copy()

    def get_network_weights_path(self):
        return self.run_folder_path / "model.pkl"

    def get_figures_path(self):
        return self.run_folder_path / "plots"

    def print_paths(self):
        print(self.run_folder_path)
        print("-\t", self.config_path)
        print("-\t", self.manifest_path)
        print("-\t", self.training_metrics_path)

    def print_last_training_metrics(self):
        print(f"After {self.config_contents.get('epochs')} epochs:")
        print(f"Loss ({self.config_contents.get('loss_type')}): {self.training_metrics.get('loss')[-1]}")
        print(f"Accuracy (RMSE): {self.training_metrics.get('accuracy')[-1]}")
        print(f"R_2: {self.training_metrics.get('r_squared')[-1]}")

    @staticmethod
    def make_run_folder_name() -> tuple[str, str]:
        now = datetime.now().strftime("%Y-%m-%d_%H-%M")
        return now, now + "_" + str(uuid.uuid4())[0:6]

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

    def output_run_files(self):
        print(f"run data saved to {self.run_folder_path}")

        self.run_folder_path.mkdir(parents=True, exist_ok=True)

        with open(self.config_path, "w") as f:
            json.dump(self.config_contents, f, indent=3)

        with open(self.manifest_path, "w") as f:
            json.dump(self.manifest_contents, f, indent=3)

        with open(self.training_metrics_path, "w") as f:
            json.dump(self.training_metrics, f, indent=3)
