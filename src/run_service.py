from argparse import Namespace
from pathlib import Path
from datetime import datetime
import hashlib
import pathlib
import json


class RunService:
    def __init__(self, args, run_dir: pathlib.Path = None):
        if run_dir is None:
            self.run_folder_name = RunService.make_run_folder_name()
            self.run_folder_path = pathlib.Path(f"runs/{self.run_folder_name}")
            self.run_folder_path.mkdir(parents=True, exist_ok=True)

            self.config_contents = {
                "model_type": args.model,
                "activation_type": args.activation,
                "loss_type": args.loss,
                "learning_rate": args.learning_rate,
                "epochs": args.epochs,
                "dataset_type": args.type,
                "dataset_size": args.dataset_size,
                "run_folder_name": self.run_folder_name,
            }
            self.config_path = self.run_folder_path / "config.json"

            self.manifest_contents = {
                "model_hash": RunService.get_model_hash(args),
                "data_hash": RunService.get_data_hash(args)
            }
            self.manifest_path = self.run_folder_path / "manifest.json"

            self.training_metrics_path = self.run_folder_path / "training.json"
            self.training_metrics = None

        else:
            raise RuntimeError("Loading prior runs not yet implemented.")

    def set_training_metrics(self, training_metrics: dict):
        self.training_metrics = training_metrics.copy()
    
    def get_network_weights_path(self):
        return self.run_folder_path / "model.pkl"
    
    def get_figures_path(self):
        return self.run_folder_path / "plots"

    @staticmethod
    def make_run_folder_name() -> str:
        return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

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
        with open(self.config_path, "w") as f:
            json.dump(self.config_contents, f, indent=3)

        with open(self.manifest_path, "w") as f:
            json.dump(self.manifest_contents, f, indent=3)
        
        with open(self.training_metrics_path, "w") as f:
            json.dump(self.training_metrics, f, indent=3)

