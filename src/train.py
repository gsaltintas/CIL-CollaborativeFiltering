import logging

from models.baselines import (
    run_baseline_training,
    run_grid_search,
    run_optuna_baselines,
)
from models.glocal_k import run_optuna_glocal_k, train_glocal_k
from utils import script_init_common

logging.basicConfig(format="%(process)d-%(levelname)s-%(message)s")
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    config = script_init_common()
    if config.experiment_type == "optuna":
        if config.algo == "svd" or config.algo == "svdpp":
            run_optuna_baselines()
        elif config.algo == "glocal_k":
            run_optuna_glocal_k()
        else:
            raise ValueError(
                f"Unknown algorithm for experiment-{config.experiment_type}, available: svd, svdpp, glocal_k"
            )
    elif config.experiment_type == "gridsearch":
        run_grid_search()
    elif config.experiment_type == "train":
        if config.algo in ["svd", "svdpp", "coclustering", "nmf", "knn", "s1"]:
            run_baseline_training()
        elif config.algo == "glocal_k":
            train_glocal_k()
        else:
            raise ValueError(
                f"Unknown algorithm for experiment-{config.experiment_type}, available: svd, svdpp, coclustering, nmf, knn, s1, glocal_k"
            )
    else:
        raise ValueError(
            f"Unknown experiment type: {config.experiment_type}, available: optuna, gridsearch, train"
        )
