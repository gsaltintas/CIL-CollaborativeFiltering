import time
from pathlib import Path

import pytorch_lightning as pl
import torch
import wandb
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

from utils import DATA_PATH, Config, script_init_common

from .data import CILDataLoader
from .finetuning import GLocalKFine
from .pretraining import GLocalKPre
from .submit import submit

config = Config()


def train_glocal_k():
    model_pre = f'nhid-{config.n_hid}-ndim--{config.n_dim}-layers-{config.n_layers}-lambda2-{config.lambda_2}-lambdas-{config.lambda_s}-iterp-{config.iter_p}-iterf-{config.iter_f}-gk-{config.gk_size}-epochp-{config.epoch_p}-epochf-{config.epoch_f}-dots-{config.dot_scale}_'
    model_dir = Path(config.experiment_dir, f'{model_pre}/{time.time():.0f}')
    model_dir.mkdir(exist_ok=True, parents=True)
    model_dir.joinpath('results').mkdir()
    model_dir = model_dir.as_posix()
    print(f'Starting model training with following configuration: {model_pre}')
    config.override('experiment_dir', model_dir)
    
    cil_dataloader = CILDataLoader(DATA_PATH, config.NUM_WORKERS)
    n_m, n_u, train_r, train_m, test_r, test_m = next(iter(cil_dataloader))

    if config.use_wandb:
        wandb.init(
            project="cil-project",
            entity="gsaltintas",
            settings=wandb.Settings(start_method="fork")
        )
        wandb_logger = WandbLogger(
            project="cil-project",
            log_model=False,
            entity="gsaltintas",
        )
        logger = wandb_logger
        wandb.config.update(config.get_all_key_values())
        print("wandb initialized")
    else:
        logger = TensorBoardLogger(
            save_dir=config.experiment_dir, log_graph=True)

    glocal_k_pre = GLocalKPre(
        config.n_hid,
        config.n_dim,
        config.n_layers,
        config.lambda_2,
        config.lambda_s,
        config.iter_p,
        n_u,
        lr=config.lr_pre,
        optim=config.optimizer,
    )
    # glocal_k_pre.double()
    pretraining_checkpoint = pl.callbacks.ModelCheckpoint(
        dirpath=f"{config.experiment_dir}/checkpoints",
        filename="pretraining-{epoch}-{pre_train_rmse:.4f}-{pre_test_rmse:.4f}",
        monitor="pre_test_rmse",
        save_top_k=2,
        mode="min",
        save_last=True,
    )
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='epoch')

    
    pretraining_trainer = pl.Trainer(
        callbacks=[pretraining_checkpoint, lr_monitor],
        max_epochs=config.epoch_p,
        log_every_n_steps=1,
        replace_sampler_ddp=False,
        logger=logger,
        accelerator='gpu' if torch.cuda.is_available() else None,
        gpus=1 if torch.cuda.is_available() else None,

    )
    pretraining_trainer.fit(glocal_k_pre, cil_dataloader, cil_dataloader)

    glocal_k_fine = GLocalKFine(
        config.gk_size,
        config.iter_f,
        config.dot_scale,
        n_m,
        pretraining_checkpoint.best_model_path,
        lr=config.lr_fine,
        optim=config.optimizer,
    )
    # glocal_k_fine.double()
    finetuning_checkpoint = pl.callbacks.ModelCheckpoint(
        dirpath=f"{config.experiment_dir}/checkpoints",
        filename="finetuning-{epoch}-{fine_train_rmse:.4f}-{fine_test_rmse:.4f}",
        monitor="fine_test_rmse",
        save_top_k=2,
        mode="min",
        save_last=True,
    )
    finetuning_trainer = pl.Trainer(
        callbacks=[finetuning_checkpoint, lr_monitor],
        max_epochs=config.epoch_f,
        log_every_n_steps=1,
        replace_sampler_ddp=False,
        logger=logger,
        accelerator='gpu' if torch.cuda.is_available() else None,
        gpus=1 if torch.cuda.is_available() else None,

    )
    finetuning_trainer.fit(glocal_k_fine, cil_dataloader, cil_dataloader)
    # glocal_k_fine = GLocalKFine.load_from_checkpoint(finetuning_checkpoint.best_model_path)
    glocal_k_fine.eval()
    pred = glocal_k_fine(train_r)
    submit(DATA_PATH, pred.detach().numpy(), Path(
        config.experiment_dir, 'results').as_posix())


if __name__ == "__main__":
    config = script_init_common()
    train_glocal_k()
