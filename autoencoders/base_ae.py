import torch
import torch.nn as nn
import lightning as pl


class AutoEncoder(pl.LightningModule):
    """
    A simple Autoencoder model implemented with PyTorch Lightning.
    It learns to compress and reconstruct a 24-dimensional vector.
    """

    def __init__(self, learning_rate=1e-3, weight_decay=1e-5, learning_patience=10):
        super().__init__()
        # This saves the hyperparameters (like learning_rate) to the checkpoint
        # and makes them accessible via self.hparams
        self.save_hyperparameters()

        self.encoder = nn.Sequential(
            nn.Linear(24, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
            nn.ReLU(),
            nn.Linear(4, 1),  # The bottleneck layer
        )
        self.decoder = nn.Sequential(
            nn.Linear(1, 4),
            nn.ReLU(),
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 24),  # Reconstructs the original 24-dim vector
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
        # The dataloader in this example yields the tensor directly.
        # If it were a tuple like (data, label), you would use: x, _ = batch
        x = batch[0]
        x_hat = self(x)  # Generate reconstruction
        loss = self.criterion(x_hat, x)
        return loss

    def training_step(self, batch, batch_idx):
        """
        Defines one step of training.
        """
        loss = self._common_step(batch, batch_idx)
        # self.log handles everything: averaging over the epoch, saving the value,
        # and sending it to the logger (in this case, the CSVLogger).
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

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

        # This scheduler reduces the learning rate if the validation loss stops improving.
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",  # We want to minimize the validation loss
            factor=0.1,  # Reduce LR by a factor of 10
            patience=self.hparams.learning_patience,  # Wait for 2 epochs of no improvement before reducing
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",  # The metric for the scheduler to watch
            },
        }
