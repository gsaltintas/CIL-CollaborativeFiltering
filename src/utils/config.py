import os
import pprint
import zipfile
from pathlib import Path
from time import time
from typing import Union

import yaml
from optuna import Study


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(
                Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class Config(object, metaclass=Singleton):
    # experiment
    project = "cil-project"
    entity = "meowtrix-purrdiction"
    experiment_dir = "."  # "/cluster/home/galtintas/scratch/"
    experiment_type = "train"  # "optuna", "train", "gridsearch"
    algo = "svd"  # options: "svd", "svdpp", "glocal_k"
    use_wandb = False
    seed = 1234 
    train_size = 0.9  # train-val split ratio

    # optuna
    n_trials = 100
    timeout = None
    verbose = 2
    enable_pruning = False
    pruner = "median"
    n_jobs = 1
    direction = "minimize"
    scoring = "rmse"
    n_startup_trials = 5
    n_warump_steps = 20
    study_name = ""
    use_storage = False  # storage for optuna
    
    # grid search
    refit = True

    # baselines
    n_factors = 66
    biased = True
    lr_all = 0.014
    reg_all = 0.09
    init_mean = 0.0
    init_std_dev = 0.1
    n_epochs = 100
    # co-cluster:
    n_cltr_u = 4
    n_cltr_i = 10
    # nmf
    reg_pu = 0.06
    reg_qi = 0.02
    # knn
    k = 25
    sim_options_name = "pearson_baseline"
    sim_options_shrinkage = "0"
    bsl_options_name = "als"

    # glocal config
    NUM_WORKERS = 8
    n_hid = 1000
    n_dim = 5
    n_layers = 3
    gk_size = 5
    lambda_2 = 20.  # l2 regularisation
    lambda_s = 0.006
    iter_p = 5  # optimization
    iter_f = 5
    epoch_p = 30
    epoch_f = 80
    dot_scale = 1.0  # scaled dot product
    lr_pre = 0.1
    lr_fine = 1.0
    optimizer = 'lbfgs'     # lbfgs, adam
    lr_scheduler = 'none'       # exponential, reducelronplateau, none
    weight_decay = 0.

    def __new__(cls):
        __instance = super().__new__(cls)
        cls.__filecontents = cls.__get_config_file_contents()
        cls.__immutable = True
        return __instance

    def import_yaml(self, yaml_path, strict=True):
        """Import YAML config to over-write existing config entries."""
        assert os.path.isfile(yaml_path)
        assert not hasattr(self.__class__, "__imported_yaml_path")
        with open(yaml_path, "r") as f:
            yaml_string = f.read()
        self.import_dict(
            yaml.load(yaml_string, Loader=yaml.FullLoader), strict=strict)
        self.__class__.__imported_yaml_path = yaml_path
        self.__class__.__filecontents[os.path.basename(
            yaml_path)] = yaml_string

    def override(self, key, value):
        self.__class__.__immutable = False
        setattr(self, key, value)
        self.__class__.__immutable = True

    def import_dict(self, dictionary, strict=True):
        """Import a set of key-value pairs from a dict to over-write existing config entries."""
        self.__class__.__immutable = False
        for key, value in dictionary.items():
            if strict is True:
                if not hasattr(self, key):
                    raise ValueError("Unknown configuration key: " + key)
                if type(getattr(self, key)) is float and type(value) is int:
                    value = float(value)
                else:
                    if getattr(self, key) is not None:
                        assert type(getattr(self, key)) is type(
                            value
                        ), f"{key}, {type(getattr(self,key))}, {type(value)}"
                if not isinstance(getattr(self, key), property):
                    setattr(self, key, value)
            else:
                if hasattr(Config, key):
                    if not isinstance(getattr(self, key), property):
                        setattr(self, key, value)
                else:
                    setattr(self, key, value)
        self.__class__.__immutable = True

    def __get_config_file_contents():
        """Retrieve and cache default and user config file contents."""
        out = {}
        for relpath in ["config.py"]:
            path = os.path.relpath(os.path.dirname(__file__) + "/" + relpath)
            assert os.path.isfile(path)
            with open(path, "r") as f:
                out[os.path.basename(path)] = f.read()
        return out

    def get_all_key_values(self):
        return dict(
            [
                (key, getattr(self, key))
                for key in dir(self)
                if not key.startswith("_Config")
                and not key.startswith("_")
                and not callable(getattr(self, key))
            ]
        )

    def get_full_yaml(self):
        return yaml.dump(self.get_all_key_values())

    def write_file_contents(self, target_base_dir: Union[Path, str]):
        """Write cached config file contents to target directory."""
        target_base_dir = Path(target_base_dir)
        assert target_base_dir.exists()

        # Write config file contents
        target_dir = target_base_dir.joinpath("configs")
        if not os.path.isdir(target_dir):
            os.makedirs(target_dir)
        outputs = {  # Also output flattened config
            "combined.yaml": self.get_full_yaml(),
        }
        outputs.update(self.__class__.__filecontents)
        for fname, content in outputs.items():
            fpath = os.path.relpath(target_dir.joinpath(fname))
            with open(fpath, "w") as f:
                f.write(content)

        # Copy source folder contents over
        target_path = os.path.relpath(target_base_dir.joinpath("src.zip"))
        source_path = os.path.relpath(os.path.dirname(__file__) + "/../")

        def filter_(x):
            return x.endswith(".py") or x.endswith(".yaml")  # noqa

        with zipfile.ZipFile(target_path, "w", zipfile.ZIP_DEFLATED) as zip_file:
            for root, dirs, files in os.walk(source_path):
                for file_or_dir in files + dirs:
                    full_path = os.path.join(root, file_or_dir)
                    if os.path.isfile(full_path) and filter_(full_path):
                        zip_file.write(
                            os.path.join(root, file_or_dir),
                            os.path.relpath(
                                os.path.join(root, file_or_dir),
                                os.path.join(source_path, os.path.pardir),
                            ),
                        )

    def __setattr__(self, name, value):
        """Initial configs should not be overwritten!"""
        if self.__class__.__immutable:
            raise AttributeError("Config instance attributes are immutable.")
        else:
            super().__setattr__(name, value)

    def __delattr__(self, name):
        """Initial configs should not be removed!"""
        if self.__class__.__immutable:
            raise AttributeError("Config instance attributes are immutable.")
        else:
            super().__delattr__(name)

    def __eq__(self, other):
        if isinstance(other, Config):
            other = other.get_all_key_values()
        elif type(other) != dict:
            raise Exception("Config can only be compared to Config or dict")
        return self.get_all_key_values() == other

    def __str__(self) -> str:
        return pprint.pformat(
            self.get_all_key_values(), indent=4, width=78, compact=True
        )

    def _get_group_dict(self, group) -> dict:
        return {key: getattr(self, key) for key in getattr(self, group)}
