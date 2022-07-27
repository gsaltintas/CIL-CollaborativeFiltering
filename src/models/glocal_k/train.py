import time
from pathlib import Path
from shutil import copy

import pytorch_lightning as pl
import wandb
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

from utils import DATA_PATH, Config, script_init_common

from .data import CILDataLoader
from .finetuning import GLocalKFine
from .pretraining import GLocalKPre
from .submit import submit

config = Config()


def train_glocal_k():
    n_hid = config.n_hid  # 1000 #500
    n_dim = config.n_dim  # 5
    n_layers = config.n_layers  # 2 #3
    gk_size = config.gk_size  # 5 #7
    lambda_2 = config.lambda_2  # 20.  # l2 regularisation
    lambda_s = config.lambda_s  # 0.006
    iter_p = config.iter_p  # 5  # optimisation
    iter_f = config.iter_f  # 5
    epoch_p = config.epoch_p  # 30
    epoch_f = config.epoch_f  # 80
    dot_scale = config.dot_scale  # 1  # scaled dot product
    seed = config.seed  # 1234
    lr_pre = config.lr_pre  # 1e-1
    lr_fine = config.lr_fine  # 1e-1
    iter_p = config.iter_p  # 5
    iter_f = config.iter_f  # 5
    dot_scale = config.dot_scale  # 1
    # setup model directory
    model_pre = f"nhid-{n_hid}-ndim--{n_dim}-layers-{n_layers}-lambda2-{lambda_2}-lambdas-{lambda_s}-iterp-{iter_p}-iterf-{iter_f}-gk-{gk_size}-epochp-{epoch_p}-epochf-{epoch_f}-dots-{dot_scale}_"
    model_dir = Path(config.experiment_dir, f"{model_pre}/{time.time():.0f}")
    model_dir.mkdir(exist_ok=True, parents=True)
    model_dir.joinpath("results").mkdir()
    model_dir = model_dir.as_posix()

    print(f"Starting model training with following configuration: {model_pre}")
    config.override("experiment_dir", model_dir)
    print(config)
    # exit
    cil_dataloader = CILDataLoader(DATA_PATH, config.NUM_WORKERS)
    n_m, n_u, train_r, train_m, test_r, test_m = next(iter(cil_dataloader))

    if config.use_wandb:
        wandb.init(
            project=config.project,
            entity=config.entity,
            settings=wandb.Settings(start_method="fork"),
        )
        wandb_logger = WandbLogger(
            project=config.project,
            log_model=False,
            entity=config.entity,
        )
        logger = wandb_logger
        wandb.config.update(config.get_all_key_values())
        print("wandb initialized")
    else:
        logger = TensorBoardLogger(save_dir=config.experiment_dir, log_graph=True)

    glocal_k_pre = GLocalKPre(
        n_hid,
        n_dim,
        n_layers,
        lambda_2,
        lambda_s,
        iter_p,
        n_u,
        lr=lr_pre,
        optim=config.optimizer,
    )
    pretraining_checkpoint = pl.callbacks.ModelCheckpoint(
        dirpath=f"{config.experiment_dir}/checkpoints",
        filename="pretraining-{epoch}-{pre_train_rmse:.4f}-{pre_test_rmse:.4f}",
        monitor="pre_test_rmse",
        save_top_k=2,
        mode="min",
        save_last=True,
    )
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval="epoch")
    pretraining_trainer = pl.Trainer(
        callbacks=[pretraining_checkpoint, lr_monitor],
        max_epochs=epoch_p,
        log_every_n_steps=1,
        replace_sampler_ddp=False,
        logger=logger,
    )
    pretraining_trainer.fit(glocal_k_pre, cil_dataloader, cil_dataloader)
    pre_ckpt = f"{config.experiment_dir}/checkpoints/pre_last.ckpt"
    copy(pretraining_checkpoint.last_model_path, pre_ckpt)
    pre_ckpt = f"{config.experiment_dir}/checkpoints/pre_best.ckpt"
    copy(pretraining_checkpoint.best_model_path, pre_ckpt)

    glocal_k_fine = GLocalKFine(
        gk_size,
        iter_f,
        dot_scale,
        n_m,
        pre_ckpt,
        lr=lr_fine,
        optim=config.optimizer,
    )
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
        max_epochs=epoch_f,
        log_every_n_steps=1,
        replace_sampler_ddp=False,
        logger=logger,
    )
    finetuning_trainer.fit(glocal_k_fine, cil_dataloader, cil_dataloader)
    glocal_k_fine = GLocalKFine.load_from_checkpoint(
        finetuning_checkpoint.best_model_path
    )
    glocal_k_fine.eval()
    pred = glocal_k_fine(train_r)
    submit(
        DATA_PATH,
        pred.detach().numpy(),
        Path(config.experiment_dir, "results").as_posix(),
    )


if __name__ == "__main__":
    config = script_init_common()
    train_glocal_k()
