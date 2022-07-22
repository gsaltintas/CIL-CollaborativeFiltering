import time
from pathlib import Path

import optuna
import pytorch_lightning as pl
import torch
import wandb
from optuna.integration import PyTorchLightningPruningCallback
from optuna.integration.wandb import WeightsAndBiasesCallback
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

from utils import DATA_PATH, DB_PATH, Config, script_init_common

from .data import CILDataLoader
from .finetuning import GLocalKFine
from .pretraining import GLocalKPre
from .submit import submit

config = Config()


def objective(trial: optuna.trial.Trial, cil_dataloader: CILDataLoader) -> float:
    print('objective')
    n_m, n_u, train_r, train_m, test_r, test_m = next(iter(cil_dataloader))

    # set up params from optuna
    # 'n_hid', [300, 350, 400, 450, 500, 550, 600])
    n_hid = trial.suggest_categorical('n_hid', [500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500])
    n_dim = config.n_dim # trial.suggest_int('n_dim', 3, 7)
    n_layers = trial.suggest_int('n_layers', 2, 5)
    lambda_2 = config.lambda_2 # trial.suggest_int('lambda_2', 10, 80)
    lambda_s = config.lambda_s # trial.suggest_float('lambda_s', 1e-3, 3e-2)
    iter_p = config.iter_p #trial.suggest_categorical('iter_p', [5*i for i in range(1, 11)])
    iter_f = config.iter_f #trial.suggest_categorical('iter_f', [5*i for i in range(1, 11)])
    gk_size = config.gk_size #trial.suggest_categorical('gk_size', [3, 5, 7, 11])
    # trial.suggest_categorical('epoch_p',[ 15, 20, 25, 30, 35, 40, 45])
    epoch_p = config.epoch_p
    # trial.suggest_categorical('epoch_f',[ 5*i for i in range(1,11)])
    epoch_f = config.epoch_f
    dot_scale = config.dot_scale # trial.suggest_float('dot_scale', 9e-1, 1.2)
    # trial.suggest_categorical('lr_pre', [1e-2, 1e-1, 1e0, 1e1, 1e2])
    lr_pre = config.lr_pre
    # trial.suggest_categorical('lr_fine', [1e-2, 1e-1, 1e0, 1e1, 1e2])
    lr_fine = config.lr_fine

    for name, val in zip(['n_hid', 'n_dim', 'n_layers', 'lambda_2', 'lambda_s', 'iter_p', 'iter_f', 'gk_size', 'epoch_p', 'epoch_f', 'dot_scale'], [n_hid, n_dim, n_layers, lambda_2, lambda_s, iter_p, iter_f, gk_size, epoch_p, epoch_f, dot_scale]):
        config.override(name, val)
    model_pre = f'trial-{trial._trial_id}-nhid-{n_hid}-ndim-{n_dim}-layers-{n_layers}-lambda2-{lambda_2}-lambdas-{lambda_s}-iterp-{iter_p}-iterf-{iter_f}-gk-{gk_size}-epochp-{epoch_p}-epochf-{epoch_f}-dots-{dot_scale}_'
    print(f'Starting model training with following configuration: {model_pre}')
    model_dir = Path(config.experiment_dir, f'{model_pre}/{time.time():.0f}')
    model_dir.mkdir(exist_ok=True, parents=True)
    model_dir.joinpath('results').mkdir()
    model_dir = model_dir.as_posix()
    config.override('experiment_dir', model_dir)

    if config.use_wandb:
        wandb.join()
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
        n_hid,
        n_dim,
        n_layers,
        lambda_2,
        lambda_s,
        iter_p,
        n_u,
        lr=lr_pre,
        trial=trial,
    )
    pretraining_checkpoint = pl.callbacks.ModelCheckpoint(
        dirpath=f"{config.experiment_dir}/checkpoints",
        filename="pretraining-{epoch}-{pre_train_rmse:.4f}-{pre_test_rmse:.4f}",
        monitor="pre_test_rmse",
        save_top_k=1,
        mode="min",
        save_last=True,
    )
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='epoch')
    callbacks_pre=[pretraining_checkpoint, lr_monitor]
    if config.enable_pruning:
        pre_pruning_callback = PyTorchLightningPruningCallback(
            trial, "pre_test_rmse")
        callbacks_pre.append(pre_pruning_callback)
    print('starting pretraining')
    pretraining_trainer = pl.Trainer(
        callbacks=callbacks_pre,
        max_epochs=epoch_p,
        log_every_n_steps=1,
        replace_sampler_ddp=False,
        gpus=1 if torch.cuda.is_available() else None,
        accelerator='gpu' if torch.cuda.is_available() else None,
        logger=logger,
    )
    pretraining_trainer.fit(glocal_k_pre, cil_dataloader, cil_dataloader)

    glocal_k_fine = GLocalKFine(
        gk_size,
        iter_f,
        dot_scale,
        n_m,
        pretraining_checkpoint.last_model_path,
        lr=lr_fine,
        trial=trial,
    )
    finetuning_checkpoint = pl.callbacks.ModelCheckpoint(
        dirpath=f"{config.experiment_dir}/checkpoints",
        filename="finetuning-{epoch}-{fine_train_rmse:.4f}-{fine_test_rmse:.4f}",
        monitor="fine_test_rmse",
        save_top_k=1,
        mode="min",
        save_last=True,
    )
    # fine_lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='epoch')
    callbacks_fine=[finetuning_checkpoint, lr_monitor]
    if config.enable_pruning:
        fine_pruning_callback = PyTorchLightningPruningCallback(
            trial, "fine_test_rmse")
        callbacks_fine.append(fine_pruning_callback)
    finetuning_trainer = pl.Trainer(
        callbacks=callbacks_fine,
        max_epochs=epoch_f,
        log_every_n_steps=1,
        replace_sampler_ddp=False,
        gpus=1 if torch.cuda.is_available() else None,
        accelerator='gpu' if torch.cuda.is_available() else None,
        logger=logger,
    )
    finetuning_trainer.fit(glocal_k_fine, cil_dataloader, cil_dataloader)

    # glocal_k_fine = GLocalKFine.load_from_checkpoint(finetuning_checkpoint.best_model_path)
    glocal_k_fine.eval()
    pred = glocal_k_fine(train_r)
    submit(DATA_PATH, pred.detach().numpy(), Path(
        config.experiment_dir, 'results').as_posix(), model_pre)
    if config.use_wandb:
        # wandb.finish()
        wandb.join()
    return finetuning_trainer.callback_metrics['test_rmse'].item()
    

class Objective(object):
    def __init__(self, cil_dataloader:CILDataLoader):
        # Hold this implementation specific arguments as the fields of the class.
        self.cil_dataloader = cil_dataloader

    def __call__(self, trial):
        return objective(trial, self.cil_dataloader)

def run_optuna_glocal_k():

    pruner: optuna.pruners.BasePruner = (
        optuna.pruners.MedianPruner(n_startup_trials=config.n_startup_trials,
                                    n_warmup_steps=config.n_warump_steps,
                                    ) if config.enable_pruning else optuna.pruners.NopPruner()
    )
    storage = None
    if config.use_storage:
        storage = optuna.storages.RDBStorage(
            url=f"sqlite:///{DB_PATH}",
            engine_kwargs={"connect_args": {"timeout": 10}},  # "pool_size": 20,
        )
    if config.study_name == "":
        config.override('study_name', f'{config.algo}-min-1')
    study = optuna.create_study(
        direction="minimize",
        study_name=f"cil-project/{config.study_name}",
        storage=storage,
        load_if_exists=True,
        pruner=pruner,
    )
    cil_dataloader = CILDataLoader(DATA_PATH, config.NUM_WORKERS)
    study.optimize(
        Objective(cil_dataloader),
        n_trials=config.n_trials,
        timeout=config.timeout,
        n_jobs=config.n_jobs,
    )
    trials = study.trials
    df = study.trials_dataframe()
    df.to_csv(config.experiment_dir + "/trials.csv")

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))


if __name__ == "__main__":
    config = script_init_common()
    run_optuna_glocal_k()
