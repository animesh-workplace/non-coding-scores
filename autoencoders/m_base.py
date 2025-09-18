import torch
import torch.nn as nn
import lightning as pl


class BigAutoEncoder(pl.LightningModule):
    """
    An enhanced Autoencoder model with BatchNorm, Dropout, and LeakyReLU activation.
    Architecture: 24→64→32→16→8→4→2→1 (bottleneck) →2→4→8→16→32→64→24
    Uses combined MSE + L1 loss for reconstruction.
    """

    def __init__(
        self,
        learning_rate=1e-3,
        weight_decay=1e-5,
        learning_patience=10,
        dropout_rate=0.2,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Linear(24, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(dropout_rate),
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(dropout_rate),
            nn.Linear(16, 8),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(dropout_rate),
            nn.Linear(8, 4),
            nn.BatchNorm1d(4),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(dropout_rate),
            nn.Linear(4, 2),
            nn.BatchNorm1d(2),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(dropout_rate),
            nn.Linear(2, 1),  # Bottleneck - no activation
        )

        # Decoder layers
        self.decoder = nn.Sequential(
            nn.Linear(1, 2),
            nn.BatchNorm1d(2),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(dropout_rate),
            nn.Linear(2, 4),
            nn.BatchNorm1d(4),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(dropout_rate),
            nn.Linear(4, 8),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(dropout_rate),
            nn.Linear(8, 16),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(dropout_rate),
            nn.Linear(16, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(dropout_rate),
            nn.Linear(32, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 24),  # Output - no activation
        )

        # Define the loss function once for efficiency.
        self.criterion = nn.L1Loss()

    def forward(self, x):
        """The forward pass of the model."""
        z = self.encoder(x)
        reconstruction = self.decoder(z)
        return reconstruction

    def _common_step(self, batch, batch_idx):
        """
        A helper function to avoid code duplication in training_step and validation_step.
        """
        x = batch[0]
        x_hat = self(x)
        loss = self.criterion(x_hat, x)
        return loss

    def training_step(self, batch, batch_idx):
        """
        Defines one step of training.
        """
        loss = self._common_step(batch, batch_idx)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def on_train_epoch_end(self):
        """
        Log the learning rate at the end of each training epoch.
        """
        current_lr = self.optimizers().param_groups[0]["lr"]

        self.log(
            "learning_rate", current_lr, prog_bar=True, on_step=False, on_epoch=True
        )

    def validation_step(self, batch, batch_idx):
        """
        Defines one step of validation.
        """
        loss = self._common_step(batch, batch_idx)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        """
        Sets up the optimizer and an optional learning rate scheduler.
        """
        optimizer = torch.optim.NAdam(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            patience=self.hparams.learning_patience,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }
