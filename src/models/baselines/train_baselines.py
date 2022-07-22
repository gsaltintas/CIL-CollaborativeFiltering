import math
import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from surprise import (
    NMF,
    SVD,
    CoClustering,
    Dataset,
    KNNBaseline,
    KNNBasic,
    Reader,
    SlopeOne,
    SVDpp,
)
from surprise.model_selection import GridSearchCV

from utils import DATA_PATH, Config, script_init_common

config = Config()

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
    res_ls = []
    for u, m in sub_data:
        pred = model.predict(uid=u, iid=m)
        res_ls.append(pred)
    return res_ls

def get_data():
    # get data
    data_pd = pd.read_csv(DATA_PATH + "data_train.csv")
    sub_pd = pd.read_csv(DATA_PATH + "sampleSubmission.csv")
    # Split the dataset into train and test
    train_size = config.train_size

    train_pd, valid_pd = train_test_split(
        data_pd, train_size=train_size, random_state=config.seed
    )

    train_users, train_movies, train_predictions = extract_users_items_predictions(
        train_pd
    )  # use whole data bc doing gridsearchcv
    train_df = pd.DataFrame()
    train_df["users"] = train_users
    train_df["movies"] = train_movies
    train_df["ratings"] = train_predictions
    data = Dataset.load_from_df(train_df, Reader(rating_scale=(1, 5)))
    data = data.build_full_trainset()
    
    valid_users, valid_movies, valid_predictions = extract_users_items_predictions(
        valid_pd
    )  
    valid_df = pd.DataFrame()
    valid_df["users"] = valid_users
    valid_df["movies"] = valid_movies
    valid_df["ratings"] = valid_predictions


    train_users, train_movies, train_predictions = extract_users_items_predictions(
        data_pd
    )  # use whole data bc doing gridsearchcv
    valid_data = Dataset.load_from_df(valid_df, Reader(rating_scale=(1, 5)))
    valid_data = valid_data.build_full_trainset()

    sub_users, sub_movies, sub_preds_wrong = extract_users_items_predictions(sub_pd)

    sub_data = zip(sub_users, sub_movies)
    return data, valid_data, sub_data, sub_users, sub_movies



def run_baseline_training():
    data, valid_data, sub_data, sub_users, sub_movies = get_data()
    if config.algo == "svd":
        params = {
            "n_factors": config.n_factors,
            "n_epochs": config.n_epochs,
            "biased": config.biased,
            "lr_all": config.lr_all,
            "reg_all": config.reg_all,
            "random_state": config.seed,
            "init_mean": config.init_mean,
            "init_std_dev":config.init_std_dev
        }

        algo = SVD(**params)
    elif config.algo == "svdpp":
        params = {
            "n_factors": config.n_factors,
            "n_epochs": config.n_epochs,
            "lr_all": config.lr_all,
            "reg_all": config.reg_all,
            "random_state": config.seed,
            "init_mean": config.init_mean,
            "init_std_dev":config.init_std_dev
        }
        algo = SVDpp(**params)
    elif config.algo == "coclustering":
        params = {
        "n_cltr_u": config.n_cltr_u,
        "n_cltr_i": config.n_cltr_i,
        "n_epochs": config.n_epochs,
        "random_state": config.seed,
    }
        algo = CoClustering(**params)
    elif config.algo == "nmf":
        params = {
            "n_factors": config.n_factors,
            "n_epochs": config.n_epochs,
            "biased":  config.biased,
            "reg_pu":config.reg_pu,
            "reg_qi": config.reg_qi,
            "random_state": config.seed,
        }
        algo = NMF(**params)
    elif config.algo == "knn":
        params ={
        "k": config.k,
        "sim_options": {"name": config.sim_options_name, "shrinkage":config.sim_options_shrinkage},
        "bsl_options": {"name": config.bsl_options_name},
    }
        algo = KNNBaseline(**params)
    elif config.algo == 's1':
        algo = SlopeOne()
    else:
        raise ValueError("Unkown algo, available: svd, svdpp, coclustering, nmf, knn, s1")
    algo.fit(data)

    sub_preds = do_preds(algo, sub_data)
    svd_df = pd.DataFrame()
    svd_df["users"] = sub_users
    svd_df["movies"] = sub_movies
    svd_df["preds"] = sub_preds
    svd_df.to_csv(f"./results/{config.algo}preds.csv", index=False)
    print(f"Saved predictions for {config.algo}")


if __name__ == "__main__":
    config = script_init_common()
    run_baseline_training()