import pytorch_lightning as pl

from data import CILDataLoader
from pretraining import GLocalKPre
from finetuning import GLocalKFine
from submit import submit

DATA_PATH = '../../../data/'
NUM_WORKERS = 8

n_hid = 500
n_dim = 5
n_layers = 3
gk_size = 7
lambda_2 = 20.  # l2 regularisation
lambda_s = 0.006
iter_p = 5  # optimisation
iter_f = 5
epoch_p = 30
epoch_f = 60
dot_scale = 1  # scaled dot product

if __name__ == '__main__':
    cil_dataloader = CILDataLoader(DATA_PATH, NUM_WORKERS)
    n_m, n_u, train_r, train_m, test_r, test_m = next(iter(cil_dataloader))

    glocal_k_pre = GLocalKPre(n_hid, n_dim, n_layers,
                              lambda_2, lambda_s, iter_p, n_u)
    pretraining_checkpoint = pl.callbacks.ModelCheckpoint(
        dirpath='checkpoints',
        filename='pretraining-{epoch}-{train_rmse:.4f}-{test_rmse:.4f}',
        monitor='test_rmse',
        save_top_k=2,
        mode='min',
        save_last=True
    )
    pretraining_trainer = pl.Trainer(callbacks=[pretraining_checkpoint],
                                     max_epochs=epoch_p,
                                     log_every_n_steps=1,
                                     replace_sampler_ddp=False
                                     )
    pretraining_trainer.fit(glocal_k_pre, cil_dataloader, cil_dataloader)

    glocal_k_fine = GLocalKFine(
        gk_size, iter_f, dot_scale, n_m, pretraining_checkpoint.last_model_path)
    finetuning_checkpoint = pl.callbacks.ModelCheckpoint(
        dirpath='checkpoints',
        filename='finetuning-{epoch}-{train_rmse:.4f}-{test_rmse:.4f}',
        monitor='test_rmse',
        save_top_k=2,
        mode='min',
        save_last=True
    )
    finetuning_trainer = pl.Trainer(callbacks=[finetuning_checkpoint],
                                    max_epochs=epoch_f,
                                    log_every_n_steps=1,
                                    replace_sampler_ddp=False
                                    )
    finetuning_trainer.fit(glocal_k_fine, cil_dataloader, cil_dataloader)

    pred = glocal_k_fine(train_r)
    submit(DATA_PATH, pred.detach().numpy())
