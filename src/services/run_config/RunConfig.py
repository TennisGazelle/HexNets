"""Run ``config.json`` shape for train-from-template and ingest normalization."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from argparse import Namespace
from copy import copy
from typing import Any, ClassVar, Mapping

from data.dataset import (
    list_registered_dataset_display_names,
    validate_run_dataset_block,
)
from networks.activation.activations import get_available_activation_functions
from networks.HexagonalNetwork import HexagonalNeuralNetwork
from networks.learning_rate.learning_rate import get_available_learning_rates
from networks.loss.loss import get_available_loss_functions
from networks.MLPNetwork import MLPNetwork
from utils import read_json_object


class RunConfig:
    """Validation and CLI merge for on-disk run ``config.json`` (train-from-file and resume ingest)."""

    INGEST_REQUIRED_KEYS: tuple[str, ...] = (
        "model_type",
        "model_metadata",
        "loss_type",
        "activation_type",
    )

    TRAIN_FALLBACK_DEFAULTS: dict[str, Any] = {
        "pause": 0.05,
        "dry_run": False,
        "dataset_noise": None,
        "dataset_noise_mu": 0.0,
        "dataset_noise_sigma": 0.1,
    }

    _override_parser: ClassVar[argparse.ArgumentParser | None] = None

    def __init__(self, raw: dict[str, Any]) -> None:
        self._raw = raw

    @staticmethod
    def validate_cli_sources(args: Namespace) -> None:
        """Ensure ``--run-config`` / ``--run-config-json`` / ``--run-dir`` combinations are valid."""
        path = getattr(args, "run_config", None)
        raw = getattr(args, "run_config_json", None)
        if path is not None and raw is not None:
            raise ValueError("Use only one of --run-config or --run-config-json, not both.")
        if (path is not None or raw is not None) and getattr(args, "run_dir", None):
            raise ValueError("Cannot combine --run-dir (resume) with --run-config or --run-config-json.")
        if path is not None:
            p = pathlib.Path(path)
            if not p.is_file():
                raise ValueError(f"--run-config is not an existing file: {p.resolve()}")

    @staticmethod
    def train_subcommand_argv(argv: list[str] | None = None) -> list[str]:
        if argv is None:
            argv = sys.argv
        try:
            i = argv.index("train")
        except ValueError:
            return []
        return list(argv[i + 1 :])

    @staticmethod
    def _learning_rate_for_namespace(config: Mapping[str, Any]) -> str | float:
        lr = config.get("learning_rate", "constant")
        if isinstance(lr, (int, float)):
            return "constant"
        return lr

    @classmethod
    def from_path(cls, path: pathlib.Path) -> RunConfig:
        cfg = read_json_object(path, "run config template")
        if not isinstance(cfg, dict):
            raise ValueError("Run config template must be a JSON object.")
        return cls(cfg)

    @classmethod
    def from_json_string(cls, raw: str) -> RunConfig:
        try:
            cfg = json.loads(raw)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON for --run-config-json: {e}") from e
        if not isinstance(cfg, dict):
            raise ValueError("--run-config-json must decode to a JSON object.")
        return cls(cfg)

    @classmethod
    def normalize_disk_config(cls, config: dict) -> None:
        """Ensure ``config['dataset']`` exists (mutates ``config`` for legacy runs)."""
        if isinstance(config.get("dataset"), dict):
            if "noise" not in config["dataset"]:
                config["dataset"]["noise"] = None
            return
        if "dataset_type" in config and "dataset_size" in config:
            config["dataset"] = {
                "id": config["dataset_type"],
                "num_samples": config["dataset_size"],
                "scale": config.get("dataset_scale"),
                "noise": None,
            }
            return
        raise ValueError(
            "Run config is missing both a 'dataset' block and legacy 'dataset_type' / 'dataset_size'. "
            "This run may be from an incompatible tool version or a partial export."
        )

    @classmethod
    def validate_disk_config_for_ingest(cls, config: dict, run_dir: pathlib.Path) -> None:
        """Minimal checks used when loading an existing run directory (same as historical ingest)."""
        cls.normalize_disk_config(config)
        missing = [k for k in cls.INGEST_REQUIRED_KEYS if k not in config]
        if missing:
            raise ValueError(
                f"Failed to ingest run at {run_dir.resolve()}: config.json is missing required keys: {missing}. "
                "This run may be from an incompatible tool version or a partial export."
            )

    @classmethod
    def from_ingested_dict(cls, data: dict[str, Any], run_dir: pathlib.Path) -> RunConfig:
        """Normalize and validate ``data`` from disk, then wrap it (mutates ``data`` in place)."""
        cls.validate_disk_config_for_ingest(data, run_dir)
        return cls(data)

    @classmethod
    def validate_for_train_template_dict(
        cls,
        config: dict[str, Any],
        *,
        errors_prefix: str = "Run config template",
    ) -> None:
        """Full validation for ``hexnet train`` starting from a template ``config.json``."""
        fake = pathlib.Path("run-config-template")
        cls.validate_disk_config_for_ingest(config, fake)

        if config.get("model_type") == "none":
            raise ValueError(f"{errors_prefix} has model_type 'none'; train supports only 'hex' and 'mlp'.")

        for key in ("epochs", "random_seed", "dataset_type", "dataset_size"):
            if key not in config:
                raise ValueError(f"{errors_prefix} is missing required key '{key}'.")

        dataset_type = config["dataset_type"]
        dataset_size = int(config["dataset_size"])
        validate_run_dataset_block(
            config["dataset"],
            dataset_type=dataset_type,
            dataset_size=dataset_size,
            errors_prefix=errors_prefix,
        )

        model_type = config["model_type"]
        meta = config["model_metadata"]
        if not isinstance(meta, dict):
            raise ValueError(f"{errors_prefix}: model_metadata must be an object.")

        if model_type == "hex":
            HexagonalNeuralNetwork.validate_run_metadata(meta, epochs=int(config["epochs"]))
        elif model_type == "mlp":
            MLPNetwork.validate_run_metadata(meta, require_square_io_for_train=True)
        else:
            raise ValueError(f"{errors_prefix}: unsupported model_type {model_type!r}.")

    @classmethod
    def _add_override_arguments(cls, p: argparse.ArgumentParser) -> None:
        sup = argparse.SUPPRESS
        g = p.add_argument_group("hex")
        g.add_argument("-n", "--num_dims", type=int, default=sup, dest="n")
        g.add_argument("-r", "--rotation", type=int, default=sup, dest="rotation")
        g.add_argument(
            "--epr",
            "--epochs-per-rotation",
            type=int,
            default=sup,
            dest="epr",
            help="Epochs per rotation (hex only; optional).",
        )
        g.add_argument(
            "--ro",
            "--rotation-ordering",
            nargs="*",
            type=int,
            default=sup,
            dest="ro",
            help="Rotation indices 0..5 (hex only; optional; space-separated after --ro).",
        )

        g = p.add_argument_group("global")
        g.add_argument("-m", "--model", type=str, default=sup, choices=["hex", "mlp", "none"], dest="model")
        g.add_argument("-s", "--seed", type=int, default=sup, dest="seed")
        g.add_argument(
            "-a",
            "--activation",
            type=str,
            default=sup,
            choices=get_available_activation_functions(),
            dest="activation",
        )
        g.add_argument("-l", "--loss", type=str, default=sup, choices=get_available_loss_functions(), dest="loss")
        g.add_argument(
            "-lr",
            "--learning-rate",
            type=str,
            default=sup,
            choices=get_available_learning_rates(),
            dest="learning_rate",
        )

        g = p.add_argument_group("training")
        g.add_argument("-e", "--epochs", type=int, default=sup, dest="epochs")
        g.add_argument("-p", "--pause", type=float, default=sup, dest="pause")
        g.add_argument(
            "-t",
            "--type",
            choices=list_registered_dataset_display_names(),
            default=sup,
            dest="type",
        )
        g.add_argument("-ds", "--dataset-size", type=int, default=sup, dest="dataset_size")
        g.add_argument(
            "--dataset-noise",
            type=str,
            default=sup,
            choices=("inputs", "targets", "both"),
            dest="dataset_noise",
        )
        g.add_argument("--dataset-noise-mu", type=float, default=sup, dest="dataset_noise_mu")
        g.add_argument("--dataset-noise-sigma", type=float, default=sup, dest="dataset_noise_sigma")
        g.add_argument("--dry-run", default=sup, action="store_true", dest="dry_run")

        g = p.add_argument_group("run")
        g.add_argument("-rn", "--run-name", type=str, default=sup, dest="run_name")
        g.add_argument("--run-note", type=str, default=sup, dest="run_note")
        g.add_argument("--run-tags", type=str, default=sup, dest="run_tags")

    @classmethod
    def _get_override_parser(cls) -> argparse.ArgumentParser:
        if cls._override_parser is None:
            cls._override_parser = argparse.ArgumentParser(add_help=False)
            cls._add_override_arguments(cls._override_parser)
        return cls._override_parser

    @classmethod
    def parse_explicit_cli_overrides(cls, argv_tail: list[str]) -> Namespace:
        """Parse train argv; only options present on the command line appear on the namespace."""
        known, _unknown = cls._get_override_parser().parse_known_args(argv_tail)
        return known

    @classmethod
    def apply_cli_only_defaults(cls, ns: Namespace) -> None:
        for key, val in cls.TRAIN_FALLBACK_DEFAULTS.items():
            if not hasattr(ns, key):
                setattr(ns, key, val)

    @classmethod
    def from_cli_sources(cls, original: Namespace) -> RunConfig:
        path = getattr(original, "run_config", None)
        raw_json = getattr(original, "run_config_json", None)
        if path is not None:
            return cls.from_path(pathlib.Path(path))
        if raw_json is not None:
            return cls.from_json_string(raw_json)
        raise ValueError("from_cli_sources called without run_config or run_config_json")

    @property
    def contents(self) -> dict[str, Any]:
        """Mutable on-disk ``config.json`` payload (same mapping as ``json`` read/write)."""
        return self._raw

    def validate(self, *, errors_prefix: str = "Run config template") -> None:
        self.validate_for_train_template_dict(self._raw, errors_prefix=errors_prefix)

    def to_namespace(self, *, errors_prefix: str = "Run config template") -> Namespace:
        self.validate(errors_prefix=errors_prefix)
        config = self._raw

        ds = config["dataset"]
        model_type = config["model_type"]
        meta = config["model_metadata"]

        ns_dict: dict[str, Any] = {
            "model": model_type,
            "activation": config["activation_type"],
            "loss": config["loss_type"],
            "learning_rate": self._learning_rate_for_namespace(config),
            "epochs": int(config["epochs"]),
            "type": config["dataset_type"],
            "dataset_size": int(config["dataset_size"]),
            "seed": int(config["random_seed"]),
            "run_dir": None,
            "dataset_scale": ds.get("scale"),
        }

        noise = ds.get("noise")
        if noise is None:
            ns_dict["dataset_noise"] = None
            ns_dict["dataset_noise_mu"] = self.TRAIN_FALLBACK_DEFAULTS["dataset_noise_mu"]
            ns_dict["dataset_noise_sigma"] = self.TRAIN_FALLBACK_DEFAULTS["dataset_noise_sigma"]
        elif isinstance(noise, dict):
            ns_dict["dataset_noise"] = noise.get("mode")
            ns_dict["dataset_noise_mu"] = float(noise.get("mu", 0.0))
            ns_dict["dataset_noise_sigma"] = float(noise.get("sigma", 0.1))
        else:
            raise ValueError(f"{errors_prefix}: dataset.noise must be null or an object.")

        if model_type == "hex":
            ns_dict["n"] = int(meta["n"])
            ns_dict["rotation"] = int(meta["r"])
            raw_epr = meta.get("epr")
            epr_ns = None if raw_epr is None else int(raw_epr)
            ns_dict["epr"] = epr_ns
            if epr_ns is None:
                ns_dict["ro"] = None
            else:
                raw_ro = meta.get("ro")
                if raw_ro is None:
                    ns_dict["ro"] = None
                elif not isinstance(raw_ro, list):
                    raise ValueError(f"{errors_prefix}: model_metadata.ro must be a JSON array or null.")
                else:
                    ns_dict["ro"] = [int(x) for x in raw_ro]
        elif model_type == "mlp":
            ns_dict["n"] = int(meta["input_dim"])
            ns_dict["rotation"] = 0

        return Namespace(**ns_dict)

    def merged_train_namespace(
        self,
        original: Namespace,
        *,
        argv_tail: list[str] | None = None,
    ) -> Namespace:
        """
        Build effective args from this template plus explicit CLI overrides.

        Preserves ``original.run-name``, ``run_note``, ``run_tags`` unless overridden on the CLI.
        Clears template source fields and ``run_dir`` for ``RunService``.
        """
        tail = argv_tail if argv_tail is not None else self.train_subcommand_argv()
        base = self.to_namespace()
        overrides = self.parse_explicit_cli_overrides(tail)

        merged = copy(base)
        for key, value in vars(overrides).items():
            setattr(merged, key, value)

        self.apply_cli_only_defaults(merged)

        if not hasattr(merged, "run_name"):
            merged.run_name = original.run_name
        if not hasattr(merged, "run_note"):
            merged.run_note = getattr(original, "run_note", None)
        if not hasattr(merged, "run_tags"):
            merged.run_tags = getattr(original, "run_tags", None)

        merged.run_dir = None
        merged.run_config = None
        merged.run_config_json = None
        return merged
