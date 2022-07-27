from typing import Optional

import optuna
import pytorch_lightning as pl
import torch
import wandb

from .glocalk_models import GlobalKernel, global_conv
from .pretraining import GLocalKPre


class GLocalKFine(pl.LightningModule):
    def __init__(
        self,
        gk_size,
        iter_f,
        dot_scale,
        n_m,
        local_kernel_checkpoint,
        lr: float = 0.1,
        trial: Optional[optuna.trial.Trial] = None,
        optim: Optional[str] = "lbfgs",
        scheduler: Optional[str] = "none",
        *args,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.iter_f = iter_f

        self.local_kernel = GLocalKPre.load_from_checkpoint(local_kernel_checkpoint)
        self.local_kernel.mode = "train"

        self.global_kernel = GlobalKernel(n_m, gk_size, dot_scale)
        self.lr = lr
        self.trial = trial
        self.optim = optim
        self.scheduler = scheduler

    def forward(self, x):
        y_dash, _ = self.local_kernel(x)

        gk = self.global_kernel(y_dash)
        y_hat = global_conv(x, gk)

        y, _ = self.local_kernel(y_hat)

        return y

    def training_step(self, batch, batch_idx):
        _, _, train_r, train_m, _, _ = batch

        y_dash, _ = self.local_kernel(train_r)

        gk = self.global_kernel(y_dash)  # Global kernel
        y_hat = global_conv(train_r, gk)  # Global kernel-based rating matrix

        pred_f, reg_losses = self.local_kernel(y_hat)

        # L2 loss
        diff = train_m * (train_r - pred_f)
        sqE = torch.sum(diff**2) / 2
        loss_f = sqE + reg_losses

        return loss_f

    def validation_step(self, batch, batch_idx):
        _, _, train_r, train_m, test_r, test_m = batch

        pred_f = self(train_r)

        error_train = (
            train_m * (torch.clip(pred_f, 1.0, 5.0) - train_r) ** 2
        ).sum() / torch.sum(train_m)
        train_rmse = torch.sqrt(error_train)

        error = (
            test_m * (torch.clip(pred_f, 1.0, 5.0) - test_r) ** 2
        ).sum() / torch.sum(test_m)
        test_rmse = torch.sqrt(error)

        self.log("train_rmse", train_rmse)
        self.log("test_rmse", test_rmse)
        self.log("fine_train_rmse", train_rmse)
        self.log("fine_test_rmse", test_rmse)
        if self.trial is not None:
            self.trial.report(test_rmse.item(), step=self.global_step)

    def configure_optimizers(self):
        if self.optim == "adam":
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        elif self.optim == "lbfgs":
            optimizer = torch.optim.LBFGS(
                self.parameters(), max_iter=self.iter_f, history_size=10, lr=self.lr
            )
        elif self.optim == "sgd":
            optimizer = torch.optim.SGD(self.parameters(), lr=self.lr)
        else:
            raise ValueError(
                "Only adam, lbfgs, and sgd options are possible for optimizer."
            )
        if self.scheduler == "exponential":
            scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer=optimizer, gamma=0.995
            )
        elif self.scheduler == "reducelronplateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, patience=4, factor=0.5, min_lr=1e-3
            )
        elif self.scheduler == "none":
            return optimizer
        else:
            raise ValueError(f"Unkown lr scheduler: {self.scheduler}")

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "test_rmse"},
        }
