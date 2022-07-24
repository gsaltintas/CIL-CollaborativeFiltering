from typing import Optional

import optuna
import pytorch_lightning as pl
import torch

from .glocalk_models import LocalKernel


class GLocalKPre(pl.LightningModule):
    def __init__(
        self,
        n_hid,
        n_dim,
        n_layers,
        lambda_2,  # l2 regularisation
        lambda_s,
        iter_p,  # optimisation
        n_u,
        lr: float = 0.1,
        trial: Optional[optuna.trial.Trial] = None,
        optim: Optional[str] = 'lbfgs',
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()

        self.iter_p = iter_p

        self.local_kernel = LocalKernel(
            n_layers, n_u, n_hid, n_dim, torch.sigmoid, lambda_s, lambda_2)
        self.lr = lr
        self.trial = trial
        self.optim = optim

    def forward(self, x):
        return self.local_kernel(x)

    def training_step(self, batch, batch_idx):
        _, _, train_r, train_m, _, _ = batch

        pred_p, reg_losses = self(train_r)

        # L2 loss
        diff = train_m * (train_r - pred_p)
        sqE = torch.sum(diff**2) / 2
        loss_p = sqE + reg_losses
        self.log('pre_train_sqE', sqE)
        self.log('pre_train_reg_losses', reg_losses)
        self.log('pre_train_loss_p', loss_p)
        return loss_p

    def validation_step(self, batch, batch_idx):
        _, _, train_r, train_m, test_r, test_m = batch

        pred_p, _ = self(train_r)

        error_train = (train_m * (torch.clip(pred_p, 1., 5.) - train_r)
                       ** 2).sum() / torch.sum(train_m)
        train_rmse = torch.sqrt(error_train)

        error = (test_m * (torch.clip(pred_p, 1., 5.) - test_r)
                 ** 2).sum() / torch.sum(test_m)
        test_rmse = torch.sqrt(error)

        self.log('pre_train_rmse', train_rmse)
        self.log('pre_test_rmse', test_rmse)
        self.log('test_rmse', test_rmse)
        if self.trial is not None:
            self.trial.report(test_rmse.item(), step=self.global_step)

    # def configure_optimizers(self):
    #     return torch.optim.LBFGS(self.local_kernel.parameters(), max_iter=self.iter_p, history_size=10, lr=self.lr)

    def configure_optimizers(self):
        if self.optim == 'adam':
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        elif self.optim == 'lbfgs':
            optimizer = torch.optim.LBFGS(
            self.parameters(), max_iter=self.iter_p, history_size=10, lr=self.lr)
        else:
            raise ValueError('Only adam and lbfgs options are possible for optimizer.')
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=4, factor=0.5, min_lr=1e-1)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer,
                                                                     gamma=0.995)

        return {'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'monitor': 'test_rmse'
                }
                }
