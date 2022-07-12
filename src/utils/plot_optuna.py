import logging
from pathlib import Path

import matplotlib.pyplot as plt
import optuna
from optuna.visualization.matplotlib import (
    plot_contour,
    plot_optimization_history,
    plot_parallel_coordinate,
    plot_param_importances,
)

from models.baselines import SVD_, SVDpp_

from .arg_parser import script_init_common

logging.basicConfig(format="%(process)d-%(levelname)s-%(message)s")
logger = logging.getLogger(__name__)
config = script_init_common()
experiment_dir = Path(config.experiment_dir, "results", "plots").absolute()
experiment_dir.mkdir(exist_ok=True, parents=True)

storage = optuna.storages.RDBStorage(
        url=f"sqlite:///{Path(config.home).joinpath('cil', 'cil.db')}",
        engine_kwargs={"connect_args": {"timeout": 10}},  # "pool_size": 20,
    )
logger.info("Trying with optuna storage")
study = optuna.create_study(
    direction="minimize",
    study_name=f"cil-project/{config.algo}-min" if config.algo=='svd' else f"cil-project/{config.algo}-min-1",
    storage=storage,
    load_if_exists=True,
)



fig = plot_optimization_history(study)
plt.savefig(fname=f"{experiment_dir}/{config.algo}_optimization_history.svg")
fig = plot_parallel_coordinate(study)
plt.savefig(fname=f"{experiment_dir}/{config.algo}_parallel_coordinate.svg")
fig = plot_contour(study)
plt.savefig(fname=f"{experiment_dir}/{config.algo}_contour.svg")
fig = plot_param_importances(study)
plt.savefig(fname=f"{experiment_dir}/{config.algo}_param_importances.svg")
