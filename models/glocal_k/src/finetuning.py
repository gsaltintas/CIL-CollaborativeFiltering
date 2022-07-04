from tkinter import Y
import torch
import pytorch_lightning as pl

from pretraining import GLocalKPre
from models import GlobalKernel, global_conv


class GLocalKFine(pl.LightningModule):
    def __init__(
        self,
        gk_size,
        iter_f,
        dot_scale,
        n_m,
        local_kernel_checkpoint,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()

        self.iter_f = iter_f

        self.local_kernel = GLocalKPre.load_from_checkpoint(
            local_kernel_checkpoint)
        self.local_kernel.mode = 'train'

        self.global_kernel = GlobalKernel(n_m, gk_size, dot_scale)

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

        error_train = (train_m * (torch.clip(pred_f, 1., 5.) -
                       train_r) ** 2).sum() / torch.sum(train_m)
        train_rmse = torch.sqrt(error_train)

        error = (test_m * (torch.clip(pred_f, 1., 5.) - test_r)
                 ** 2).sum() / torch.sum(test_m)
        test_rmse = torch.sqrt(error)

        self.log('train_rmse', train_rmse)
        self.log('test_rmse', test_rmse)

    def configure_optimizers(self):
        return torch.optim.LBFGS(self.local_kernel.parameters(), max_iter=self.iter_f)
