import numpy as np
import pandas as pd
import surprise
from surprise import SVD, Dataset, Reader, SVDpp


def extract_users_items_predictions(data_pd):
    users, movies = [
        np.squeeze(arr)
        for arr in np.split(
            data_pd.Id.str.extract("r(\d+)_c(\d+)").values.astype(int) - 1, 2, axis=-1
        )
    ]
    predictions = data_pd.Prediction.values
    return users, movies, predictions


def get_trainset():
    data_pd = pd.read_csv("./data/data_train.csv")

    train_users, train_movies, train_predictions = extract_users_items_predictions(
        data_pd
    )  # use whole data bc doing gridsearchcv

    train_df = pd.DataFrame()
    train_df["users"] = train_users
    train_df["movies"] = train_movies
    train_df["ratings"] = train_predictions

    data = Dataset.load_from_df(train_df, Reader(rating_scale=(1, 5)))
    # required to directly call fit
    trainset = data.build_full_trainset()

    return trainset


class SVD_(SVD):
    """Wrapper class for SVD to be used with Optuna"""

    def __init__(
        self,
        trainset=None,
        n_factors=100,
        n_epochs=20,
        biased=True,
        init_mean=0,
        init_std_dev=0.1,
        lr_all=0.005,
        reg_all=0.02,
        lr_bu=None,
        lr_bi=None,
        lr_pu=None,
        lr_qi=None,
        reg_bu=None,
        reg_bi=None,
        reg_pu=None,
        reg_qi=None,
        random_state=None,
        verbose=False,
        *args,
        **kwargs
    ) -> None:
        SVD.__init__(
            self,
            n_factors,
            n_epochs,
            biased,
            init_mean,
            init_std_dev,
            lr_all,
            reg_all,
            lr_bu,
            lr_bi,
            lr_pu,
            lr_qi,
            reg_bu,
            reg_bi,
            reg_pu,
            reg_qi,
            random_state,
            verbose,
            *args,
            **kwargs
        )
        if trainset is None:
            trainset = get_trainset()
        if type(trainset) != surprise.trainset.Trainset:
            trainset = trainset.build_full_trainset()
        # self.trainset = get_trainset()
        self.trainset = trainset

    def get_params(self, deep=None):
        params_lst = [
            "trainset",
            "n_factors",
            "n_epochs",
            "biased",
            "init_mean",
            "init_std_dev",
            "lr_bu",
            "lr_bi",
            "lr_pu",
            "lr_qi",
            "reg_bu",
            "reg_bi",
            "reg_pu",
            "reg_qi",
            "random_state",
            "verbose",
        ]
        return {key: getattr(self, key) for key in params_lst}

    def set_params(self, **params):
        for key, val in params.items():
            setattr(self, key, val)

    def fit(self, X, y):
        return SVD.fit(self, self.trainset)

    def predict(self, X):
        res_ls = [SVD.predict(self, u, m).est for (u, m) in X]
        return res_ls


class SVDpp_(SVDpp):
    """Wrapper class for SVD to be used with Optuna"""

    def __init__(
        self,
        trainset=None,
        n_factors=20,
        n_epochs=20,
        init_mean=0,
        init_std_dev=0.1,
        lr_all=0.007,
        reg_all=0.02,
        lr_bu=None,
        lr_bi=None,
        lr_pu=None,
        lr_qi=None,
        lr_yj=None,
        reg_bu=None,
        reg_bi=None,
        reg_pu=None,
        reg_qi=None,
        reg_yj=None,
        random_state=None,
        verbose=False,
        *args,
        **kwargs
    ):
        SVDpp.__init__(
            self,
            n_factors,
            n_epochs,
            init_mean,
            init_std_dev,
            lr_all,
            reg_all,
            lr_bu,
            lr_bi,
            lr_pu,
            lr_qi,
            lr_yj,
            reg_bu,
            reg_bi,
            reg_pu,
            reg_qi,
            reg_yj,
            random_state,
            verbose,
            *args,
            **kwargs
        )
        self.lr_all = lr_all
        self.reg_all = reg_all
        if trainset is None:
            trainset = get_trainset()
        if type(trainset) != surprise.trainset.Trainset:
            trainset = trainset.build_full_trainset()
        self.trainset = trainset

    def get_params(self, deep=None):
        params_lst = [
            "trainset",
            "n_factors",
            "n_epochs",
            "init_mean",
            "init_std_dev",
            "lr_all",
            "reg_all",
            "lr_bu",
            "lr_bi",
            "lr_pu",
            "lr_qi",
            "lr_yj",
            "reg_bu",
            "reg_bi",
            "reg_pu",
            "reg_qi",
            "reg_yj",
            "random_state",
            "verbose",
        ]
        return {key: getattr(self, key) for key in params_lst}

    def set_params(self, **params):
        for key, val in params.items():
            setattr(self, key, val)

    def fit(self, X, y):
        return SVDpp.fit(self, self.trainset)

    def predict(self, X):
        res_ls = [SVDpp.predict(self, u, m).est for (u, m) in X]
        return res_ls
