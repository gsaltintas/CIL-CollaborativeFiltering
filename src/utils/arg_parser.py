import argparse
import logging
from pathlib import Path

import numpy as np
import torch
from pytorch_lightning.utilities.seed import seed_everything

from .config import Config

config = Config()
logger = logging.getLogger(__name__)


def _convert_cli_arg_type(key, value):
    config_type = type(getattr(Config, key))
    if config_type == bool:
        if value.lower() in ("true", "yes", "y") or value == "1":
            return True
        elif value.lower() in ("false", "no", "n") or value == "0":
            return False
        else:
            raise ValueError('Invalid input for bool config "%s": %s' % (key, value))
    else:
        return config_type(value)


def script_init_common():
    parser = argparse.ArgumentParser(
        description="CIL Project - meowtrix-purrdiction Group."
    )
    parser.add_argument(
        "-v",
        type=str,
        help="Desired logging level.",
        default="info",
        choices=["debug", "info", "warning", "error", "critical"],
    )
    parser.add_argument(
        "config_yaml",
        type=str,
        nargs="*",
        help=(
            "Path to config in YAML format. "
            "Multiple configs will be parsed in the specified order."
        ),
    )
    for key in dir(config):
        if key.startswith("_Config") or key.startswith("_"):
            continue
        if key in vars(Config) and isinstance(vars(Config)[key], property):
            continue
        value = getattr(config, key)
        value_type = type(value)
        arg_type = value_type
        nargs = None
        if value_type == bool:
            # Handle booleans separately, otherwise arbitrary values become `True`
            arg_type = str
        if value_type == list:
            # handle lists
            arg_type = str if len(value) == 0 else type(value[0])
            nargs = "*"
        if value_type == Path:
            arg_type = str
        if callable(value):
            continue
        parser.add_argument(
            "--" + key.replace("_", "-"),
            type=arg_type,
            metavar=value,
            help="Expected type is `%s`." % value_type.__name__,
            nargs=nargs,
        )
    args = parser.parse_args()
    # args, unknown = parser.parse_known_args()
    # Parse configs in order specified by user
    for yaml_path in args.config_yaml:
        config.import_yaml(yaml_path)

    # Apply configs passed through command line
    config.import_dict(
        {
            key.replace("-", "_"): _convert_cli_arg_type(key, value)
            for key, value in vars(args).items()
            if value is not None and hasattr(config, key)
        }
    )
    set_seed(config.seed)
    return config


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    seed_everything(seed=config.seed, workers=True)
