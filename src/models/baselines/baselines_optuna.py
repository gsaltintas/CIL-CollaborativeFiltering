import logging
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import surprise
import wandb
from optuna.integration.wandb import WeightsAndBiasesCallback
from sklearn.model_selection import train_test_split
from surprise import Dataset, Reader

from utils import DATA_PATH, Config, script_init_common

from .SVD import SVD_, SVDpp_

logging.basicConfig(format="%(process)d-%(levelname)s-%(message)s")
logger = logging.getLogger(__name__)
config = Config()
experiment_dir = Path(f"{config.experiment_dir}")
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
    """ 
    Uses model to predict ratings for test data
    Writes the submission file to experiment_dir/prefixsubmission.csv
    """
    submission_pd = pd.read_csv(DATA_PATH + "sampleSubmission.csv")
    submission_users, submission_movies, _ = extract_users_items_predictions(
        submission_pd
    )
    preds = model.predict(zip(submission_users, submission_movies))
    submission_pd["Prediction"] = preds
    file_name = f"{experiment_dir}/{prefix}submission.csv"
    submission_pd.to_csv(file_name, encoding="utf-8", index=False)

def get_data():
    data_pd = pd.read_csv(DATA_PATH + "data_train.csv")
    # Split the dataset into train and test
    train_size = config.train_size

    train_pd, valid_pd = train_test_split(
        data_pd, train_size=train_size, random_state=config.seed
    )

    train_users, train_movies, train_predictions = extract_users_items_predictions(
        train_pd
    ) 
    train_df = pd.DataFrame()
    train_df["users"] = train_users
    train_df["movies"] = train_movies
    train_df["ratings"] = train_predictions

    valid_users, valid_movies, valid_predictions = extract_users_items_predictions(
        valid_pd
    ) 
    valid_df = pd.DataFrame()
    valid_df["users"] = valid_users
    valid_df["movies"] = valid_movies
    valid_df["ratings"] = valid_predictions

    X = np.column_stack((train_users, train_movies))
    y = train_predictions
    X_val = np.column_stack((valid_users, valid_movies))
    y_val = valid_predictions
    return train_df, X, y, X_val, y_val


def objective(trial):
    """
    Optuna objective function, trial picks new set of parameters and fits the model
    RMSE on the validation set is returned as the objective value
    Submission file is created with the fitted model
    """
    train_df, X, y, X_val, y_val  = get_data()
    if config.algo == "svd":
        algo = SVD_
        params = {
            "n_factors": trial.suggest_int("n_factors", 50, 200),
            "biased": trial.suggest_categorical("biased", [True, False]),
            "lr_all": trial.suggest_float("lr_all", 1e-5, 0.1),
            "reg_all": trial.suggest_float("reg_all", 1e-3, 0.1),
            "random_state": config.seed,
            "init_mean": trial.suggest_float("init_mean", 0, 3),
            "init_std_dev": trial.suggest_float("init_std_dev", 0.1, 1),
        }
    elif config.algo == "svdpp":
        algo = SVDpp_
        params = {
            "n_factors": trial.suggest_int("n_factors", 40, 200),
            "lr_all": trial.suggest_float("lr_all", 1e-5, 0.1),
            "reg_all": trial.suggest_float("reg_all", 1e-3, 0.1),
            "n_epochs": trial.suggest_int("n_epochs", 40, 70),
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
        wandb.log(params, step=trial._trial_id)
    a.fit(X, y)
    score = np.mean(0.5 * np.square(a.predict(X) - y))
    logger.info(f" Train Value: {score}")
    score_val = np.mean(0.5 * np.square(a.predict(X_val) - y_val))
    logger.info(f" Validation Value: {score_val}")
    submission_file = f"{config.algo}_{trial._trial_id}_t-{score}_v-{score_val:.3f}"
    if config.use_wandb:
        wandb.define_metric("trial_id")
        wandb.log(
            {"train-rmse": score, "val-rmse": score_val, "trial_id": trial._trial_id},
            step=trial._trial_id,
        )

    write_submission(a, submission_file)
    print(f"Saved predictions for {config.algo}")

    return score_val


def run_optuna_baselines():
    """
    Run the optuna search for the baselines svd and svdpp
    saves a submission file for each trial
    """
    storage = optuna.storages.RDBStorage(
        url=f"sqlite:///{Path(config.home).joinpath('cil', 'cil.db')}",
        engine_kwargs={"connect_args": {"timeout": 10}},  # "pool_size": 20,
    )
    logger.info("Trying with optuna storage")
    study = optuna.create_study(
        direction="minimize",
        study_name=f"cil-project/{config.algo}-min"
        if config.algo == "svd"
        else f"cil-project/{config.algo}-min-2",
        storage=storage,
        load_if_exists=True,
    )
    callbacks = []
    if config.use_wandb:
        import wandb

        wandb_kwargs = {
            "project": "cil-project",
            "entity": "gsaltintas",
            "config": config.get_all_key_values(),
        }
        wandbc = WeightsAndBiasesCallback(
            metric_name=config.scoring, wandb_kwargs=wandb_kwargs
        )
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


if __name__ == "__main__":
    config = script_init_common()
    config.override("seed", np.random.randint(0, 2**32))
    run_optuna_baselines()
