import torch
import torch.nn as nn
import lightning as pl


class MaskedDenoisingAutoEncoder(pl.LightningModule):
    def __init__(
        self,
        learning_rate=1e-3,
        weight_decay=1e-5,
        learning_patience=10,
        corruption_level=0.1,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.criterion = nn.L1Loss()

        # Input dimension is now 48 (24 original features + 24 mask features)
        self.encoder = nn.Sequential(
            nn.Linear(48, 32),  # Increased from 24 to 48 input features
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),  # bottleneck
        )
        # Output dimension is still 24 (we're reconstructing the original features)
        self.decoder = nn.Sequential(
            nn.Linear(1, 8),
            nn.ReLU(),
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 24),
        )

    def forward(self, x):
        """The forward pass of the model."""
        z = self.encoder(x)
        reconstruction = self.decoder(z)
        return reconstruction

    def _common_step(self, model_input, target):
        """
        A refactored helper function.
        """
        reconstruction = self(model_input)
        loss = self.criterion(reconstruction, target)
        return loss

    def training_step(self, batch, batch_idx):
        """
        Defines one step of training with masking.
        """
        # 1. Get the original, clean data
        (x,) = batch

        # 2. Create a binary mask where True indicates a value to corrupt
        corruption_mask = torch.rand_like(x) < self.hparams.corruption_level

        # 3. Create the noisy input by setting corrupted values to 0
        x_noisy = x.clone()
        x_noisy[corruption_mask] = 0.0

        # 4. Create the binary indicator mask (1 for corrupted, 0 for intact)
        binary_mask = corruption_mask.float()

        # 5. Concatenate the noisy features with the binary mask
        model_input = torch.cat([x_noisy, binary_mask], dim=1)

        # 6. Calculate loss using the combined input and the clean target
        loss = self._common_step(model_input=model_input, target=x)

        # 7. Log the training loss
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        During validation, we measure how well the model reconstructs a CLEAN input.
        Therefore, we don't add any noise here, but we still need to provide a mask.
        Since there's no corruption, the mask will be all zeros.
        """
        (x,) = batch

        # Create a zero mask (no corruption in validation)
        zero_mask = torch.zeros_like(x)

        # Concatenate the clean features with the zero mask
        model_input = torch.cat([x, zero_mask], dim=1)

        # Calculate loss using the clean input and clean target
        loss = self._common_step(model_input=model_input, target=x)

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
