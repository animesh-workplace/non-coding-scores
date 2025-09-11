import torch
import torch.nn as nn
import lightning as pl
from torch.utils.data import TensorDataset, DataLoader
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.loggers import CSVLogger
import os


class AutoEncoder(pl.LightningModule):
    """
    A simple Autoencoder model implemented with PyTorch Lightning.
    It learns to compress and reconstruct a 24-dimensional vector.
    """

    def __init__(self, learning_rate=1e-3, weight_decay=1e-5):
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
            patience=2,  # Wait for 2 epochs of no improvement before reducing
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",  # The metric for the scheduler to watch
            },
        }


# ----------------------------------------------------------------------------
# 2. Main script to set up data and run the training
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    # --- Configuration ---
    BATCH_SIZE = 32
    MAX_EPOCHS = 20
    LEARNING_RATE = 1e-3

    # --- Data Setup (using dummy data) ---
    print("Setting up dummy data...")
    train_data = torch.randn(2000, 24)
    val_data = torch.randn(400, 24)

    # PyTorch Lightning works best with DataLoader objects
    train_loader = DataLoader(
        TensorDataset(train_data), batch_size=BATCH_SIZE, num_workers=4
    )
    val_loader = DataLoader(
        TensorDataset(val_data), batch_size=BATCH_SIZE, num_workers=4
    )

    # --- Logger and Callback Setup ---
    # The CSVLogger will save all metrics logged with self.log() to a CSV file.
    csv_logger = CSVLogger(save_dir="lightning_logs/", name="autoencoder_run")

    # The LearningRateMonitor logs the learning rate at the end of each epoch.
    # This is crucial for seeing when the scheduler changes the LR.
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    # --- Model and Trainer Initialization ---
    print("Initializing model and trainer...")
    autoencoder = AutoEncoder(learning_rate=LEARNING_RATE)

    # The Trainer orchestrates the entire training process.
    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        logger=csv_logger,  # Pass the CSVLogger instance here
        callbacks=[lr_monitor],  # Pass the LR monitor in a list of callbacks
    )

    # --- Start Training ---
    print(f"Starting training for {MAX_EPOCHS} epochs...")
    trainer.fit(
        model=autoencoder, train_dataloaders=train_loader, val_dataloaders=val_loader
    )

    print("\nTraining complete!")
    print(
        f"Metrics have been saved to: {os.path.join(csv_logger.log_dir, 'metrics.csv')}"
    )
