import logging
import time
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import surprise
from optuna.integration.wandb import WeightsAndBiasesCallback
from sklearn.model_selection import train_test_split
from surprise import Dataset, Reader
from surprise.accuracy import rmse

import wandb
from models import SVD_, SVDpp_
from utils.arg_parser import script_init_common

logging.basicConfig(format="%(process)d-%(levelname)s-%(message)s")
logger = logging.getLogger(__name__)
config = script_init_common()
experiment_dir = Path(f"results/{config.algo}_{time.time()}")
experiment_dir.mkdir(exist_ok=True, parents=True)


def extract_users_items_predictions(data_pd):
    users, movies = [
        np.squeeze(arr)
        for arr in np.split(
            data_pd.Id.str.extract("r(\d+)_c(\d+)").values.astype(int) - 1, 2, axis=-1
        )
    ]
    predictions = data_pd.Prediction.values
    return users, movies, predictions


def write_submission(model, prefix):
    submission_pd = pd.read_csv("./data/sampleSubmission.csv")
    submission_users, submission_movies, _ = extract_users_items_predictions(
        submission_pd
    )
    preds = model.predict(zip(submission_users, submission_movies))
    submission_pd["Prediction"] = preds
    file_name = f"{experiment_dir}/{prefix}submission.csv"
    submission_pd.to_csv(file_name, encoding="utf-8", index=False)


# get data
data_pd = pd.read_csv("./data/data_train.csv")
# Split the dataset into train and test
train_size = 0.9

train_pd, valid_pd = train_test_split(data_pd, train_size=train_size, random_state=config.seed)

train_users, train_movies, train_predictions = extract_users_items_predictions(
    train_pd
)  # use whole data bc doing gridsearchcv
train_df = pd.DataFrame()
train_df["users"] = train_users
train_df["movies"] = train_movies
train_df["ratings"] = train_predictions

valid_users, valid_movies, valid_predictions = extract_users_items_predictions(
    valid_pd
)  # use whole data bc doing gridsearchcv
valid_df = pd.DataFrame()
valid_df["users"] = valid_users
valid_df["movies"] = valid_movies
valid_df["ratings"] = valid_predictions

X = np.column_stack((train_users, train_movies))
y = train_predictions
X_val = np.column_stack((valid_users, valid_movies))
y_val = valid_predictions


def objective(trial):
    if config.algo == "svd":
        algo = SVD_
        params = {
            "n_factors": trial.suggest_int("n_factors", 50, 200),
            "biased": trial.suggest_categorical("biased", [True, False]),
            "lr_all": trial.suggest_float("lr_all", 1e-5, 0.1),
            "reg_all": trial.suggest_float("reg_all", 1e-3, 0.1),
            # "random_state": trial.suggest_int("random_state", 42, 42),
            "random_state": config.seed,
            "init_mean": trial.suggest_float("init_mean", 0, 3),
            "init_std_dev": trial.suggest_float("init_std_dev", 0.1, 1),
        }
    elif config.algo == "svdpp":
        algo = SVDpp_
        params = {
            "n_factors": trial.suggest_int("n_factors", 10, 200),
            "lr_all": trial.suggest_float("lr_all", 1e-5, 0.1),
            "reg_all": trial.suggest_float("reg_all", 1e-3, 0.1),
            "n_epochs": trial.suggest_int("n_epochs", 15, 25),
            # "random_state": trial.suggest_int("random_state", 42, 42),
            "random_state": config.seed,
            "init_mean": trial.suggest_float("init_mean", 0, 3),
            "init_std_dev": trial.suggest_float("init_std_dev", 0.1, 1),
        }
    else:
        raise ValueError(f"Optuna search not implemented for {config.algo}")
    data = Dataset.load_from_df(train_df, Reader(rating_scale=(1, 5)))
    if isinstance(data, surprise.dataset.DatasetAutoFolds):
        data = data.build_full_trainset()
    a = algo(trainset=data, verbose=config.verbose, **params)
    if config.use_wandb:
        wandb.define_metric("trial_id")
        wandb.log(params, step=trial.trial_id)
    a.fit(X, y)
    score = np.mean(0.5 * np.square(a.predict(X) - y))
    logger.info(f" Train Value: {score}")
    score_val = np.mean(0.5 * np.square(a.predict(X_val) - y_val))
    logger.info(f" Validation Value: {score_val}")
    submission_file = f"{trial._trial_id}_t-{score}_v-{score_val:.3f}"
    if config.use_wandb:
        wandb.define_metric("trial_id")
        wandb.log(
            {"train-rmse": score, "val-rmse": score_val, "trial_id": trial._trial_id}, step=trial.trial_id
        )
        files = wandb.config['submission_files']
        files.append(submission_file)
        wandb.config.update({'submissions_files': files})

    write_submission(a, submission_file)
    print(f"Saved predictions for {config.algo}")

    return score_val


if __name__ == "__main__":
    storage = optuna.storages.RDBStorage(
        url=f"sqlite:///{Path(config.home).joinpath('cil', 'cil.db')}",
        engine_kwargs={"connect_args": {"timeout": 10}},  # "pool_size": 20,
    )
    logger.info("Trying with optuna storage")
    study = optuna.create_study(
        direction="minimize",
        study_name=f"cil-project/{config.algo}-min",
        storage=storage,
        load_if_exists=True,
    )
    callbacks = []
    if config.use_wandb:
        wandb_kwargs = {"project": "cil-project", "entity": "gsaltintas"}
        wandbc = WeightsAndBiasesCallback(
            metric_name=config.scoring, wandb_kwargs=wandb_kwargs
        )
        callbacks.append(wandbc)
        wandb.config.update(config.get_all_key_values())
    study.optimize(
        objective,
        n_trials=config.n_trials,
        timeout=config.timeout,
        callbacks=callbacks,
        n_jobs=config.n_jobs,
    )
    # doesnt reach here within the limits of euler
    trials = study.trials
    df = study.trials_dataframe()
    if config.use_wandb:
        wandb.log({"Trials": wandb.Table(dataframe=df)})
    df.to_csv(experiment_dir.joinpath("trials.csv"))

    for trial in trials:
        logger.info({"rmse": trial.value})
        logger.info("  Params: ")
        if config.use_wandb:
            wandb.log(
                {"rmse": trial.value, "trial_id": trial._trial_id}, step=trial._trial_id
            )
        for k, v in trial.params.items():
            logger.info("    {}: {}".format(k, v))
            if config.use_wandb:
                wandb.log({f"{k}": v}, step=trial._trial_id)
            else:
                logger.info({f"{k}": v})

    trial = study.best_trial
