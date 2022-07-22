import numpy as np
import pandas as pd
import torch

from utils import Config

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


def load_data_cil(path="./", frac=(1 - config.train_size), seed=config.seed):
    data_pd = pd.read_csv(path + "data_train.csv")
    users, movies, predictions = extract_users_items_predictions(data_pd)
    data = pd.DataFrame.from_dict(
        {"userId": users, "itemId": movies, "rating": predictions}
    )

    n_u = np.unique(data["userId"]).size  # num of users
    n_m = np.unique(data["itemId"]).size  # num of movies
    n_r = data.shape[0]  # num of ratings

    udict = {}
    for i, u in enumerate(np.unique(data["userId"]).tolist()):
        udict[u] = i
    mdict = {}
    for i, m in enumerate(np.unique(data["itemId"]).tolist()):
        mdict[m] = i

    np.random.seed(seed)
    idx = np.arange(n_r)
    np.random.shuffle(idx)

    train_r = np.zeros((n_m, n_u), dtype="float32")
    test_r = np.zeros((n_m, n_u), dtype="float32")

    for i in range(n_r):
        u_id = data.loc[idx[i]]["userId"]
        m_id = data.loc[idx[i]]["itemId"]
        r = data.loc[idx[i]]["rating"]

        if i < int(frac * n_r):
            test_r[m_id, u_id] = r
        else:
            train_r[m_id, u_id] = r

    # masks indicating non-zero entries
    train_m = np.greater(train_r, 1e-12).astype("float32")
    test_m = np.greater(test_r, 1e-12).astype("float32")

    print("data matrix loaded")
    print("num of users: {}".format(n_u))
    print("num of movies: {}".format(n_m))
    print("num of training ratings: {}".format(n_r - int(frac * n_r)))
    print("num of test ratings: {}".format(int(frac * n_r)))

    return n_m, n_u, train_r, train_m, test_r, test_m


class CILDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, seed=1234):
        self.data = load_data_cil(data_path)

    def __len__(self):
        return 1

    def __getitem__(self, _):
        return self.data


class CILDataLoader(torch.utils.data.DataLoader):
    def __init__(self, data_path, num_workers):
        super().__init__(
            CILDataset(data_path), batch_size=None, num_workers=num_workers
        )
