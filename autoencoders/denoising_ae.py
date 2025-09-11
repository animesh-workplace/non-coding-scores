import torch
import torch.nn as nn
import lightning as pl


class DenoisingAutoEncoder(pl.LightningModule):
    def __init__(
        self,
        learning_rate=1e-3,
        weight_decay=1e-5,
        learning_patience=10,
        corruption_level=0.1,
        corruption_value=99999,
    ):
        super().__init__()
        # MODIFICATION: Added corruption_level and corruption_value to __init__
        self.save_hyperparameters()
        self.criterion = nn.L1Loss()

        self.encoder = nn.Sequential(
            nn.Linear(24, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
            nn.ReLU(),
            nn.Linear(4, 1),  # bottleneck
        )
        self.decoder = nn.Sequential(
            nn.Linear(1, 4),
            nn.ReLU(),
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 24),
        )

    def forward(self, x):
        """The forward pass of the model."""
        z = self.encoder(x)
        reconstruction = self.decoder(z)
        return reconstruction

    def _common_step(self, model_input, target):
        """
        A refactored helper function. It now takes the model's input and the
        loss target as separate arguments.
        """
        reconstruction = self(model_input)
        loss = self.criterion(reconstruction, target)
        return loss

    def training_step(self, batch, batch_idx):
        """
        Defines one step of training, now with the denoising logic.
        """
        # 1. Get the original, clean data
        (x,) = batch

        # 2. Create a noisy version of the input
        x_noisy = x.clone()
        # Create a boolean mask where True indicates a value to corrupt
        corruption_mask = torch.rand_like(x) < self.hparams.corruption_level
        # Apply corruption
        x_noisy[corruption_mask] = self.hparams.corruption_value

        # 3. Calculate loss using the noisy input and the clean target
        loss = self._common_step(model_input=x_noisy, target=x)

        # 4. Log the training loss
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        During validation, we measure how well the model reconstructs a CLEAN input.
        Therefore, we don't add any noise here.
        """
        (x,) = batch

        # Calculate loss using the clean input and clean target
        loss = self._common_step(model_input=x, target=x)

        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def on_train_epoch_end(self):
        """
        Log the learning rate at the end of each training epoch.
        """
        current_lr = self.optimizers().param_groups[0]["lr"]

        self.log(
            "learning_rate", current_lr, prog_bar=True, on_step=False, on_epoch=True
        )

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
