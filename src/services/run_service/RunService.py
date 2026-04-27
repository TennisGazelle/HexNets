from argparse import Namespace
from datetime import datetime
import hashlib
import pathlib
import json
import uuid
from typing import Any, List, Tuple, Union

from utils import read_json_from_path, read_json_object, resolve_git_commit

from networks.activation.activations import get_activation_function
from networks.loss.loss import get_loss_function
from networks.MLPNetwork import MLPNetwork
from networks.HexagonalNetwork import HexagonalNeuralNetwork
from services.logging_config import get_logger

logger = get_logger(__name__)


class RunService:
    runs_dir = pathlib.Path("runs/").resolve()
    _REQUIRED_CONFIG_KEYS = (
        "model_type",
        "model_metadata",
        "loss_type",
        "activation_type",
    )

    def __init__(self, args):
        def init_run():
            timestamp, run_folder_name = RunService.make_run_folder_name(
                args.run_name if args.run_name else None
            )
            self.run_folder_path = RunService.runs_dir / run_folder_name
            self.config_path = self.run_folder_path / "config.json"
            self.manifest_path = self.run_folder_path / "manifest.json"
            self.training_metrics_path = self.run_folder_path / "training_metrics.json"

            self.run_folder_path.mkdir(parents=True, exist_ok=True)

            self.loss_function = get_loss_function(args.loss)
            self.activation_function = get_activation_function(args.activation)

            ds = getattr(args, "dataset_scale", None)
            self.config_contents = {
                "schema_version": 1,
                "model_type": args.model,
                "activation_type": self.activation_function.display_name,
                "loss_type": self.loss_function.display_name,
                "learning_rate": args.learning_rate,
                "epochs": args.epochs,
                "dataset_type": args.type,
                "dataset_size": args.dataset_size,
                "dataset": {
                    "id": args.type,
                    "num_samples": args.dataset_size,
                    "scale": ds,
                },
                "random_seed": args.seed,
                "run_folder_name": self.run_folder_path.name,
                "model_metadata": {},
            }

            git_commit, git_error = resolve_git_commit()
            self.manifest_contents = {
                "schema_version": 1,
                "model_hash": RunService.get_model_hash(args),
                "data_hash": RunService.get_data_hash(args),
                "date_first_run": timestamp,
                "git_commit": git_commit,
                "random_seed": args.seed,
                "trainable_parameter_count": 0,
                "run_note": getattr(args, "run_note", None),
                "run_tags": RunService._parse_run_tags(getattr(args, "run_tags", None)),
            }
            if git_commit is None and git_error:
                self.manifest_contents["git_error"] = git_error

            self.training_metrics_contents = None

            if args.model == "mlp":
                self.config_contents["model_metadata"]["input_dim"] = args.n
                self.config_contents["model_metadata"]["hidden_dims"] = [4, 5, 4]
                self.config_contents["model_metadata"]["output_dim"] = args.n
                self.net = MLPNetwork(
                    input_dim=args.n,
                    output_dim=args.n,
                    hidden_dims=[4, 5, 4],
                    learning_rate=args.learning_rate,
                    activation=self.activation_function,
                    loss=self.loss_function,
                )
                self.manifest_contents["trainable_parameter_count"] = (
                    MLPNetwork.get_parameter_count(args.n, args.n, [4, 5, 4])
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
                self.manifest_contents["trainable_parameter_count"] = (
                    HexagonalNeuralNetwork.get_parameter_count(args.n)
                )
            else:
                raise ValueError(f"Invalid model: {args.model}")

        def load_run():
            self.run_folder_path = args.run_dir
            self.config_path = self.run_folder_path / "config.json"
            self.manifest_path = self.run_folder_path / "manifest.json"
            self.training_metrics_path = self.run_folder_path / "training_metrics.json"

            run_abs = self.run_folder_path.resolve()
            try:
                self.config_contents = read_json_object(
                    self.config_path, "run config (config.json)"
                )
            except ValueError as e:
                raise ValueError(f"Failed to ingest run at {run_abs}.\n{e}") from e
            try:
                self.manifest_contents = read_json_object(
                    self.manifest_path, "run manifest (manifest.json)"
                )
            except ValueError as e:
                raise ValueError(f"Failed to ingest run at {run_abs}.\n{e}") from e
            try:
                self.training_metrics_contents = read_json_from_path(
                    self.training_metrics_path,
                    "training metrics (training_metrics.json)",
                )
            except ValueError as e:
                raise ValueError(f"Failed to ingest run at {run_abs}.\n{e}") from e

            try:
                RunService._normalize_dataset_in_config(self.config_contents)
            except ValueError as e:
                raise ValueError(f"Failed to ingest run at {run_abs}.\n{e}") from e
            RunService._validate_loaded_run_config(
                self.config_contents, self.run_folder_path
            )

            self.loss_function = get_loss_function(self.config_contents["loss_type"])
            self.activation_function = get_activation_function(
                self.config_contents["activation_type"]
            )
            # Handle backward compatibility: if learning_rate is a float, use constant
            learning_rate_config = self.config_contents.get("learning_rate", "constant")
            if isinstance(learning_rate_config, (int, float)):
                learning_rate_config = "constant"

            # load the network
            if self.config_contents["model_type"] == "mlp":
                self.net = MLPNetwork(
                    input_dim=self.config_contents["model_metadata"]["input_dim"],
                    output_dim=self.config_contents["model_metadata"]["output_dim"],
                    hidden_dims=self.config_contents["model_metadata"]["hidden_dims"],
                    learning_rate=learning_rate_config,
                    activation=self.activation_function,
                    loss=self.loss_function,
                )

            elif self.config_contents["model_type"] == "hex":
                self.net = HexagonalNeuralNetwork(
                    n=self.config_contents["model_metadata"]["n"],
                    r=self.config_contents["model_metadata"]["r"],
                    learning_rate=learning_rate_config,
                    activation=self.activation_function,
                    loss=self.loss_function,
                )

            self.net.load(self.get_network_weights_path())

        if (
            "run_dir" not in args
            or args.run_dir is None
            or ("run_name" in args and args.run_name)
        ):
            init_run()
        else:
            load_run()

    # --- static methods ---
    @staticmethod
    def make_run_folder_name(filename: Union[str, None] = None) -> Tuple[str, str]:
        now = datetime.now().strftime("%Y-%m-%d_%H-%M")
        return now, now + "_" + str(uuid.uuid4())[0:6] if filename is None else filename

    @staticmethod
    def _normalize_dataset_in_config(config: dict) -> None:
        """Ensure ``config['dataset']`` exists (mutates ``config`` for legacy runs)."""
        if isinstance(config.get("dataset"), dict):
            return
        if "dataset_type" in config and "dataset_size" in config:
            config["dataset"] = {
                "id": config["dataset_type"],
                "num_samples": config["dataset_size"],
                "scale": config.get("dataset_scale"),
            }
            return
        raise ValueError(
            "Run config is missing both a 'dataset' block and legacy 'dataset_type' / 'dataset_size'. "
            "This run may be from an incompatible tool version or a partial export."
        )

    @staticmethod
    def _validate_loaded_run_config(config: dict, run_dir: pathlib.Path) -> None:
        missing = [k for k in RunService._REQUIRED_CONFIG_KEYS if k not in config]
        if missing:
            raise ValueError(
                f"Failed to ingest run at {run_dir.resolve()}: config.json is missing required keys: {missing}. "
                "This run may be from an incompatible tool version or a partial export."
            )

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
        return hashlib.sha256(
            f"{args.model}_{params}_{args.activation}_{args.loss}".encode()
        ).hexdigest()

    @staticmethod
    def get_data_hash(args: Namespace) -> str:
        parts: List[str] = [args.type, str(args.dataset_size)]
        scale = getattr(args, "dataset_scale", None)
        if args.type == "linear_scale" and scale is not None:
            parts.append(str(scale))
        payload = "_".join(parts)
        return hashlib.sha256(payload.encode()).hexdigest()

    @staticmethod
    def _parse_run_tags(raw: Any) -> List[str]:
        if raw is None or raw == "":
            return []
        if isinstance(raw, list):
            return [str(t).strip() for t in raw if str(t).strip()]
        return [part.strip() for part in str(raw).split(",") if part.strip()]

    # --- instance methods ---
    def set_training_metrics(self, training_metrics: dict):
        self.training_metrics_contents = training_metrics.copy()

    def get_network_weights_path(self) -> pathlib.Path:
        return self.run_folder_path / "model.pkl"

    def get_figures_path(self) -> pathlib.Path:
        return self.run_folder_path / "plots"

    def print_paths(self) -> None:
        logger.info(self.run_folder_path)
        logger.info(f"-\t{self.config_path.name}")
        logger.info(f"-\t{self.manifest_path.name}")
        logger.info(f"-\t{self.training_metrics_path.name}")

    def print_last_training_metrics(self) -> None:
        self.net.show_latest_metrics()

    def output_run_files(self) -> None:
        logger.info(f"run data saved to {self.run_folder_path}")

        self.run_folder_path.mkdir(parents=True, exist_ok=True)
        self.get_figures_path().mkdir(parents=True, exist_ok=True)

        with open(self.config_path, "w") as f:
            json.dump(self.config_contents, f, indent=3)

        with open(self.manifest_path, "w") as f:
            json.dump(self.manifest_contents, f, indent=3)

        with open(self.training_metrics_path, "w") as f:
            json.dump(self.training_metrics_contents, f, indent=3)
