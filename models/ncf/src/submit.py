import pandas as pd
import numpy as np
from psutil import users
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from gmf import GMFEngine
from mlp import MLPEngine
from neumf import NeuMFEngine
from data import SampleGenerator
from train import gmf_config, mlp_config, neumf_config
from utils import extract_users_items_predictions

NUM_WORKERS = 8
BATCH_SIZE = 256
DEVICE = 'cpu'

if __name__ == '__main__':
    submission_pd = pd.read_csv('./data/cil/sampleSubmission.csv')
    submission_users, submission_movies, _ = extract_users_items_predictions(submission_pd)
    submission_users_torch = torch.tensor(submission_users, device=DEVICE)
    submission_movies_torch = torch.tensor(submission_movies, device=DEVICE)
    # create submission dataloader
    submit_loader = DataLoader(TensorDataset(submission_users_torch, submission_movies_torch),
                            batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

    # Specify the exact model
    # config = gmf_config
    # engine = GMFEngine(config)
    # config = mlp_config
    # engine = MLPEngine(config)
    config = neumf_config
    engine = NeuMFEngine(config)

    engine.model.eval()
    with torch.no_grad():
        submission_i = 0
        for users_batch, movies_batch in iter(submit_loader):
            # print(users_batch, movies_batch)
            pred_batch = engine.model(users_batch, movies_batch)
            # print(pred_batch)
            start_ind = submission_i*BATCH_SIZE
            end_ind = pred_batch.shape[0] + start_ind
            # print(start_ind, end_ind)
            preds_cpu = pred_batch.cpu().detach().numpy()
            preds_cpu = preds_cpu * 5.0
            np.clip(preds_cpu, a_min=1, a_max=5, out=preds_cpu)
            submission_pd.iloc[start_ind:end_ind, 1] = preds_cpu
            submission_i += 1

    file_name = 'submission.csv'
    submission_pd.to_csv(file_name, encoding='utf-8', index=False)