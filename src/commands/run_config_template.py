"""Load run ``config.json`` templates for ``hexnet train`` and merge explicit CLI overrides."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from argparse import Namespace
from copy import copy
from typing import Any, Mapping

from networks.learning_rate.learning_rate import get_available_learning_rates
from networks.activation.activations import get_available_activation_functions
from networks.loss.loss import get_available_loss_functions
from data.dataset import list_registered_dataset_display_names

from services.run_service import RunService
from utils import read_json_object


def validate_run_config_cli_exclusivity(args: Namespace) -> None:
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


# Defaults mirroring ``commands.command`` (used only for CLI-only fields after template + overrides).
_TRAIN_FALLBACK_DEFAULTS: dict[str, Any] = {
    "pause": 0.05,
    "dry_run": False,
    "dataset_noise": None,
    "dataset_noise_mu": 0.0,
    "dataset_noise_sigma": 0.1,
}


def train_subcommand_argv(argv: list[str] | None = None) -> list[str]:
    """Return argv tokens after the ``train`` subcommand."""
    if argv is None:
        argv = sys.argv
    try:
        i = argv.index("train")
    except ValueError:
        return []
    return list(argv[i + 1 :])


def load_run_config_template_from_path(path: pathlib.Path) -> dict[str, Any]:
    """Read and validate a JSON object from disk."""
    cfg = read_json_object(path, "run config template")
    if not isinstance(cfg, dict):
        raise ValueError("Run config template must be a JSON object.")
    return cfg


def load_run_config_template_from_string(raw: str) -> dict[str, Any]:
    """Parse a JSON object from a string."""
    try:
        cfg = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON for --run-config-json: {e}") from e
    if not isinstance(cfg, dict):
        raise ValueError("--run-config-json must decode to a JSON object.")
    return cfg


def validate_run_config_template(config: dict[str, Any]) -> None:
    """Normalize dataset block and enforce the same required keys as ingesting a run directory."""
    RunService._normalize_dataset_in_config(config)
    fake = pathlib.Path("run-config-template")
    RunService._validate_loaded_run_config(config, fake)


def _learning_rate_for_namespace(config: Mapping[str, Any]) -> str | float:
    lr = config.get("learning_rate", "constant")
    if isinstance(lr, (int, float)):
        return "constant"
    return lr


def run_config_to_namespace(config: dict[str, Any]) -> Namespace:
    """Map a validated on-disk config dict to ``TrainCommand`` / ``RunService`` namespace fields."""
    validate_run_config_template(config)

    if config.get("model_type") == "none":
        raise ValueError("Run config template has model_type 'none'; train supports only 'hex' and 'mlp'.")

    for key in ("epochs", "random_seed", "dataset_type", "dataset_size"):
        if key not in config:
            raise ValueError(f"Run config template is missing required key '{key}'.")

    ds = config["dataset"]
    if not isinstance(ds, dict):
        raise ValueError("Run config template: 'dataset' must be an object after normalization.")

    dataset_type = config["dataset_type"]
    if ds.get("id") != dataset_type:
        raise ValueError(
            "Run config template: dataset.id must match dataset_type "
            f"(got id={ds.get('id')!r}, dataset_type={dataset_type!r})."
        )
    if ds.get("num_samples") != config["dataset_size"]:
        raise ValueError(
            "Run config template: dataset.num_samples must match dataset_size "
            f"(got num_samples={ds.get('num_samples')!r}, dataset_size={config['dataset_size']!r})."
        )

    model_type = config["model_type"]
    meta = config["model_metadata"]
    if not isinstance(meta, dict):
        raise ValueError("Run config template: model_metadata must be an object.")

    ns_dict: dict[str, Any] = {
        "model": model_type,
        "activation": config["activation_type"],
        "loss": config["loss_type"],
        "learning_rate": _learning_rate_for_namespace(config),
        "epochs": int(config["epochs"]),
        "type": dataset_type,
        "dataset_size": int(config["dataset_size"]),
        "seed": int(config["random_seed"]),
        "run_dir": None,
        "dataset_scale": ds.get("scale"),
    }

    noise = ds.get("noise")
    if noise is None:
        ns_dict["dataset_noise"] = None
        ns_dict["dataset_noise_mu"] = _TRAIN_FALLBACK_DEFAULTS["dataset_noise_mu"]
        ns_dict["dataset_noise_sigma"] = _TRAIN_FALLBACK_DEFAULTS["dataset_noise_sigma"]
    elif isinstance(noise, dict):
        mode = noise.get("mode")
        if mode not in ("inputs", "targets", "both"):
            raise ValueError(
                "Run config template: dataset.noise.mode must be one of inputs, targets, both when noise is set."
            )
        ns_dict["dataset_noise"] = mode
        ns_dict["dataset_noise_mu"] = float(noise.get("mu", 0.0))
        ns_dict["dataset_noise_sigma"] = float(noise.get("sigma", 0.1))
    else:
        raise ValueError("Run config template: dataset.noise must be null or an object.")

    if model_type == "hex":
        if "n" not in meta or "r" not in meta:
            raise ValueError("Run config template: hex model_metadata requires 'n' and 'r'.")
        ns_dict["n"] = int(meta["n"])
        ns_dict["rotation"] = int(meta["r"])
    elif model_type == "mlp":
        for k in ("input_dim", "output_dim", "hidden_dims"):
            if k not in meta:
                raise ValueError(f"Run config template: mlp model_metadata requires '{k}'.")
        if int(meta["input_dim"]) != int(meta["output_dim"]):
            raise ValueError(
                "Run config template: MLP training expects input_dim == output_dim (matches CLI --num_dims)."
            )
        ns_dict["n"] = int(meta["input_dim"])
        ns_dict["rotation"] = 0
    else:
        raise ValueError(f"Run config template: unsupported model_type {model_type!r}.")

    return Namespace(**ns_dict)


def _add_override_arguments(p: argparse.ArgumentParser) -> None:
    """Same destinations as train-related flags, all ``SUPPRESS`` so only explicit argv sets them."""
    sup = argparse.SUPPRESS
    g = p.add_argument_group("hex")
    g.add_argument("-n", "--num_dims", type=int, default=sup, dest="n")
    g.add_argument("-r", "--rotation", type=int, default=sup, dest="rotation")

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
    g.add_argument("-rn", "--run_name", type=str, default=sup, dest="run_name")
    g.add_argument("--run-note", type=str, default=sup, dest="run_note")
    g.add_argument("--run-tags", type=str, default=sup, dest="run_tags")


_override_parser: argparse.ArgumentParser | None = None


def _get_override_parser() -> argparse.ArgumentParser:
    global _override_parser
    if _override_parser is None:
        _override_parser = argparse.ArgumentParser(add_help=False)
        _add_override_arguments(_override_parser)
    return _override_parser


def parse_explicit_cli_overrides(argv_tail: list[str]) -> Namespace:
    """Parse argv for train; only options present on the command line appear on the namespace."""
    known, _unknown = _get_override_parser().parse_known_args(argv_tail)
    return known


def apply_cli_only_defaults(ns: Namespace) -> None:
    """Fill pause / dry_run / noise defaults when absent (CLI-only fields not in config.json)."""
    for key, val in _TRAIN_FALLBACK_DEFAULTS.items():
        if not hasattr(ns, key):
            setattr(ns, key, val)


def merge_run_config_template_args(original: Namespace) -> Namespace:
    """
    Build effective args from ``--run-config`` / ``--run-config-json`` plus explicit CLI overrides.

    Preserves ``original.run_name``, ``run_note``, ``run_tags`` unless overridden on the CLI.
    Clears template source fields and ``run_dir`` for ``RunService``.
    """
    path = getattr(original, "run_config", None)
    raw_json = getattr(original, "run_config_json", None)
    if path is not None:
        cfg = load_run_config_template_from_path(pathlib.Path(path))
    elif raw_json is not None:
        cfg = load_run_config_template_from_string(raw_json)
    else:
        raise ValueError("merge_run_config_template_args called without run_config or run_config_json")

    base = run_config_to_namespace(cfg)
    overrides = parse_explicit_cli_overrides(train_subcommand_argv())

    merged = copy(base)
    for key, value in vars(overrides).items():
        setattr(merged, key, value)

    # CLI-only defaults not supplied by template or explicit flag
    apply_cli_only_defaults(merged)

    # Carry forward run identity fields unless overridden above
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
