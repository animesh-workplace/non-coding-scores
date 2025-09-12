import torch
import torch.nn as nn
import lightning as pl
import numpy as np


class OrthogonalAutoEncoder(pl.LightningModule):
    def __init__(
        self,
        learning_rate=1e-3,
        weight_decay=1e-5,
        learning_patience=10,
        corruption_level=0.1,
        latent_dim=4,
        ortho_lambda=0.1,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.criterion = nn.L1Loss()
        self.latent_dim = latent_dim

        # self.encoder = nn.Sequential(
        #     nn.Linear(48, 32),  # 24 features + 24 mask indicators
        #     nn.ReLU(),
        #     nn.Linear(32, 16),
        #     nn.ReLU(),
        #     nn.Linear(16, 8),
        #     nn.ReLU(),
        #     nn.Linear(8, latent_dim),  # Multiple latent dimensions
        # )
        self.encoder = nn.Sequential(
            nn.Linear(48, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Linear(8, latent_dim),
            # Typically no BatchNorm on the final latent layer
        )

        # self.decoder = nn.Sequential(
        #     nn.Linear(latent_dim, 8),
        #     nn.ReLU(),
        #     nn.Linear(8, 16),
        #     nn.ReLU(),
        #     nn.Linear(16, 32),
        #     nn.ReLU(),
        #     nn.Linear(32, 24),  # Reconstruct original 24 features
        # )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 8),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Linear(8, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 24),
            # Typically no BatchNorm on the final output layer
        )

    def orthogonal_regularization(self, x):
        # Get latent representations for batch
        z = self.encoder(x)

        # Compute correlation matrix
        z_normalized = (z - z.mean(0)) / (z.std(0) + 1e-8)
        correlation = torch.mm(z_normalized.t(), z_normalized) / z.size(0)

        # Measure deviation from identity matrix (perfect orthogonality)
        identity = torch.eye(self.latent_dim, device=z.device)
        ortho_loss = torch.norm(correlation - identity, p="fro")

        return ortho_loss

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
        Defines one step of training with masking and orthogonal regularization.
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

        # 6. Calculate reconstruction loss
        recon_loss = self._common_step(model_input=model_input, target=x)

        # 7. Calculate orthogonal regularization loss
        ortho_loss = self.orthogonal_regularization(model_input)

        # 8. Combined loss
        total_loss = recon_loss + self.hparams.ortho_lambda * ortho_loss

        # 9. Log the losses
        self.log("train_loss", total_loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("train_recon_loss", recon_loss, on_step=False, on_epoch=True)
        self.log("train_ortho_loss", ortho_loss, on_step=False, on_epoch=True)

        return total_loss

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
        recon_loss = self._common_step(model_input=model_input, target=x)

        # Calculate orthogonal regularization loss
        ortho_loss = self.orthogonal_regularization(model_input)

        # Combined loss
        total_loss = recon_loss + self.hparams.ortho_lambda * ortho_loss

        self.log("val_loss", total_loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_recon_loss", recon_loss, on_step=False, on_epoch=True)
        self.log("val_ortho_loss", ortho_loss, on_step=False, on_epoch=True)

        return total_loss

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

    def analyze_latent_dimensions(self, X_tensor, score_cols, df):
        """
        Analyze what each latent dimension represents by computing correlations
        with the original features.
        """
        self.eval()
        device = next(self.parameters()).device

        # Create a zero mask for inference
        zero_mask = torch.zeros_like(X_tensor)
        model_input = torch.cat([X_tensor, zero_mask], dim=1).to(device)

        with torch.no_grad():
            latent = self.encoder(model_input).cpu().numpy()

        # Check correlation with original features
        results = {}
        for dim in range(self.latent_dim):
            correlations = {}
            for i, col in enumerate(score_cols):
                corr = np.corrcoef(df[col].values, latent[:, dim])[0, 1]
                correlations[col] = corr

            # Store top features for this dimension
            top_features = sorted(
                correlations.items(), key=lambda x: abs(x[1]), reverse=True
            )[:5]
            results[f"dim_{dim}"] = {
                "correlations": correlations,
                "top_features": top_features,
            }

        return results
