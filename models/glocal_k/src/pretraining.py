import torch
import pytorch_lightning as pl

from models import LocalKernel


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
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()

        self.iter_p = iter_p

        self.local_kernel = LocalKernel(
            n_layers, n_u, n_hid, n_dim, torch.sigmoid, lambda_s, lambda_2)

    def forward(self, x):
        return self.local_kernel(x)

    def training_step(self, batch, batch_idx):
        _, _, train_r, train_m, _, _ = batch

        pred_p, reg_losses = self(train_r)

        # L2 loss
        diff = train_m * (train_r - pred_p)
        sqE = torch.sum(diff**2) / 2
        loss_p = sqE + reg_losses

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

        self.log('train_rmse', train_rmse)
        self.log('test_rmse', test_rmse)

    def configure_optimizers(self):
        return torch.optim.LBFGS(self.local_kernel.parameters(), max_iter=self.iter_p)
