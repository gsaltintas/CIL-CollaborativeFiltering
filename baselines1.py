import logging
import math
import multiprocessing
import os
import sys
import time
from cgitb import enable

import numpy as np
import optuna
import pandas as pd
import surprise
from joblib import Parallel, delayed
from optuna.integration.wandb import WeightsAndBiasesCallback
from optuna.trial import TrialState
from surprise import NMF, SVD, Dataset, KNNBaseline, KNNBasic, Reader, SVDpp
from surprise.model_selection import GridSearchCV

import wandb
from models.optuna.SVD import SVD_, SVDpp_
from utils.arg_parser import script_init_common

logging.basicConfig(format="%(process)d-%(levelname)s-%(message)s")
logger = logging.getLogger(__name__)
config = script_init_common()


def extract_users_items_predictions(data_pd):
    users, movies = [
        np.squeeze(arr)
        for arr in np.split(
            data_pd.Id.str.extract("r(\d+)_c(\d+)").values.astype(int) - 1, 2, axis=-1
        )
    ]
    predictions = data_pd.Prediction.values
    return users, movies, predictions


def do_preds(model, sub_data):
    try:
        cpus = multiprocessing.cpu_count()
    except NotImplementedError:
        cpus = 2  # arbitrary default
    res_ls = Parallel(n_jobs=cpus * 2)(
        delayed(model.predict)((u, m)) for (u, m) in sub_data
    )
    res_ls = [res.est for res in res_ls]
    return res_ls


def run_svd(trial, data, sub_data, sub_users, sub_movies):
    # init svd
    logger.info("running svd")
    algo = SVD_
    param_dict = {
        "n_factors": optuna.distributions.IntUniformDistribution(50, 200),
        "biased": optuna.distributions.CategoricalDistribution([True, False]),
        "lr_all": optuna.distributions.UniformDistribution(1e-5, 0.1),
        "reg_all": optuna.distributions.UniformDistribution(1e-3, 0.1),
        "random_state": optuna.distributions.IntUniformDistribution(42, 42),
    }
    X = np.column_stack((data["train_users"], data["train_movies"]))
    y = data[""]
    if isinstance(data, surprise.dataset.DatasetAutoFolds):
        data = data.build_full_trainset()
    optuna_search = optuna.integration.OptunaSearchCV(
        algo(),
        param_dict,
        n_trials=config.n_trials,
        timeout=config.timeout,
        verbose=config.verbose,
        enable_pruning=config.enable_pruning,
    )
    optuna_search.fit(data)
    if config.use_wandb:
        wandb.log({"trials": wandb.Table(dataframe=optuna_search.trials_dataframe)})
    else:
        optuna_search.trials_dataframe.to_csv(config.dir.joinpath("svd.csv"))

    print("Best trial:")
    trial = optuna_search.study_.best_trial

    print("  Value: ", trial.value)
    print("  Params: ")
    for k, v in trial.params.items():
        print("    {}: {}".format(k, v))
        if config.use_wandb:
            wandb.log({f"Best/{k}": v})
        else:
            logger.info({f"Best/{k}": v})

    sub_preds = do_preds(trial, sub_data)
    svd_df = pd.DataFrame()
    svd_df["users"] = sub_users
    svd_df["movies"] = sub_movies
    svd_df["preds"] = sub_preds
    filename = config.dir.joinpath("SVD_preds.csv")
    svd_df.to_csv(filename, index=False)
    print("Saved predictions for SVD")


def run_svdpp(trial, data, sub_data, sub_users, sub_movies):
    # init svd
    print("running svdpp")
    algo = SVDpp_
    param_dict = {
        "n_factors": optuna.distributions.IntUniformDistribution(15, 25),
        "lr_all": optuna.distributions.UniformDistribution(1e-5, 0.1),
        "reg_all": optuna.distributions.UniformDistribution(1e-3, 0.1),
        "random_state": optuna.distributions.IntUniformDistribution(42, 42),
    }
    X = np.column_stack((data["train_users"], data["train_movies"]))
    y = data[""]
    if type(data) == surprise.dataset.DatasetAutoFolds:
        data = data.build_full_trainset()
    optuna_search = optuna.integration.OptunaSearchCV(
        algo(), param_dict, n_trials=100, timeout=600, verbose=2
    )

    optuna_search.fit(data)
    print("Best trial:")
    trial = optuna_search.study_.best_trial

    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    sub_preds = do_preds(trial, sub_data)
    svd_df = pd.DataFrame()
    svd_df["users"] = sub_users
    svd_df["movies"] = sub_movies
    svd_df["preds"] = sub_preds
    svd_df.to_csv(f"./results/SVDpreds{time.time():.0f}.csv", index=False)
    print("Saved predictions for SVD")


def objective(tria):
    # get data
    data_pd = pd.read_csv("./data/data_train.csv")
    sub_pd = pd.read_csv("./data/sampleSubmission.csv")

    train_users, train_movies, train_predictions = extract_users_items_predictions(
        data_pd
    )  # use whole data bc doing gridsearchcv

    train_df = pd.DataFrame()
    train_df["users"] = train_users
    train_df["movies"] = train_movies
    train_df["ratings"] = train_predictions

    data = Dataset.load_from_df(train_df, Reader(rating_scale=(1, 5)))

    sub_users, sub_movies, sub_preds_wrong = extract_users_items_predictions(sub_pd)

    sub_data = zip(sub_users, sub_movies)
    if config.algo == "SVD":
        run_svd(trial, data, sub_data, sub_users, sub_movies)
    elif config.algo == "svdpp":
        run_svdpp(trial, data, sub_data, sub_users, sub_movies)
    else:
        raise ValueError(f"Optuna search not implemented for {config.algo}")


if __name__ == "__main__":
    study = optuna.create_study(
        direction="minimize", study_name="cil-project/baselines"
    )
    # run selected model
    cmdline = sys.argv[1]  # arg 0 is name of file
    callbacks = []
    if config.use_wandb:
        wandb_kwargs = {"project": "cil-project", "entity": "gsaltintas"}
        wandbc = WeightsAndBiasesCallback(metric_name="rmse", wandb_kwargs=wandb_kwargs)
        callbacks.append(wandbc)

    study.optimize(
        objective,
        n_trials=config.n,
        # timeout=,
        callbacks=callbacks,
    )
    #   if (cmdline == 'svd'):
    #         run_svd_single(data, sub_data, sub_users, sub_movies)
    #   elif (cmdline == 'nmf'):
    #     run_nmf_single(data, sub_data, sub_users, sub_movies)
    #   else:
    # #     run_knn(data, sub_data, sub_users, sub_movies)
    # pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    # complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    # print("Study statistics: ")
    # print("  Number of finished trials: ", len(study.trials))
    # print("  Number of pruned trials: ", len(pruned_trials))
    # print("  Number of complete trials: ", len(complete_trials))

    # print("Best trial:")
    # trial = study.best_trial

    # print("  Value: ", trial.value)

    # print("  Params: ")
    # for key, value in trial.params.items():
    #     print("    {}: {}".format(key, value))
