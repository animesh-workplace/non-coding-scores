import torch
import torch.nn as nn
import lightning as pl


class SinkhornDistance(nn.Module):
    """
    Approximate Wasserstein distance using the Sinkhorn algorithm.
    This is a differentiable approximation of the Wasserstein distance.
    """

    def __init__(self, eps=0.01, max_iter=100, reduction="mean"):
        super(SinkhornDistance, self).__init__()
        self.eps = eps
        self.max_iter = max_iter
        self.reduction = reduction

    def forward(self, x, y):
        # x and y are of shape (batch_size, n_features)
        # We'll compute the distance between each pair of samples
        # and then average across the batch

        # Compute the cost matrix (Euclidean distance between samples)
        x_col = x.unsqueeze(-2)  # (batch_size, 1, n_features)
        y_lin = y.unsqueeze(-3)  # (1, batch_size, n_features)
        C = (x_col - y_lin).pow(2).sum(-1)  # (batch_size, batch_size)

        # Sinkhorn algorithm
        batch_size = x.size(0)
        mu = torch.ones(batch_size, device=x.device) / batch_size
        nu = torch.ones(batch_size, device=y.device) / batch_size

        u = torch.zeros_like(mu)
        v = torch.zeros_like(nu)

        # To avoid in-place operations & ensure differentiability
        for _ in range(self.max_iter):
            v = (
                self.eps
                * (
                    torch.log(nu)
                    - torch.logsumexp((u.unsqueeze(1) - C) / self.eps, dim=0)
                )
                + v
            )
            u = (
                self.eps
                * (
                    torch.log(mu)
                    - torch.logsumexp((v.unsqueeze(0) - C) / self.eps, dim=1)
                )
                + u
            )

        # Compute the transport plan and distance
        T = torch.exp((u.unsqueeze(1) + v.unsqueeze(0) - C) / self.eps)
        distance = torch.sum(T * C)

        if self.reduction == "mean":
            return distance / batch_size
        else:
            return distance


class WassersteinAutoEncoder(pl.LightningModule):
    """
    Autoencoder with Wasserstein-inspired loss function.
    """

    def __init__(
        self,
        learning_rate=1e-3,
        weight_decay=1e-5,
        learning_patience=10,
        use_wasserstein=True,
        wasserstein_weight=0.1,
        feature_weights=None,
    ):
        super().__init__()
        self.save_hyperparameters()

        # If no feature weights are provided, use equal weights
        if feature_weights is None:
            feature_weights = [1.0] * 24

        self.encoder = nn.Sequential(
            nn.Linear(24, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
            nn.ReLU(),
            nn.Linear(4, 2),
        )
        self.decoder = nn.Sequential(
            nn.Linear(2, 4),
            nn.ReLU(),
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 24),
        )

        # Standard reconstruction loss
        self.reconstruction_loss = nn.L1Loss()

        # Wasserstein distance approximation
        self.use_wasserstein = use_wasserstein
        if use_wasserstein:
            self.sinkhorn = SinkhornDistance(eps=0.1, max_iter=100)

        # Feature weights for weighted reconstruction loss
        self.register_buffer(
            "feature_weights", torch.tensor(feature_weights, dtype=torch.float32)
        )
        self.wasserstein_weight = wasserstein_weight

    def forward(self, x):
        """The forward pass of the model."""
        z = self.encoder(x)
        reconstruction = self.decoder(z)
        return reconstruction, z

    def _compute_loss(self, x, x_hat):
        """Compute the combined loss."""
        # Standard reconstruction loss (weighted)
        recon_loss = torch.mean(torch.abs(x_hat - x) * self.feature_weights)

        # Add Wasserstein distance if enabled
        wasserstein_loss = 0
        if self.use_wasserstein:
            # Reshape for Sinkhorn algorithm (treat each feature as a separate distribution)
            wasserstein_loss = 0
            for i in range(x.size(1)):
                # Compute Wasserstein distance for each feature separately
                wasserstein_loss += self.sinkhorn(
                    x[:, i].unsqueeze(1), x_hat[:, i].unsqueeze(1)
                )
            wasserstein_loss /= x.size(1)  # Average across features

            total_loss = recon_loss + self.wasserstein_weight * wasserstein_loss
        else:
            total_loss = recon_loss

        return total_loss, recon_loss, wasserstein_loss

    def _common_step(self, batch, batch_idx):
        """A helper function for training and validation steps."""
        x = batch[0]
        x_hat, z = self(x)
        loss, recon_loss, wasserstein_loss = self._compute_loss(x, x_hat)
        return loss, recon_loss, wasserstein_loss

    def training_step(self, batch, batch_idx):
        """Defines one step of training."""
        loss, recon_loss, wasserstein_loss = self._common_step(batch, batch_idx)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("train_recon_loss", recon_loss, on_step=False, on_epoch=True)
        if self.use_wasserstein:
            self.log(
                "train_wasserstein_loss", wasserstein_loss, on_step=False, on_epoch=True
            )
        return loss

    def on_train_epoch_end(self):
        """Log the learning rate at the end of each training epoch."""
        current_lr = self.optimizers().param_groups[0]["lr"]
        self.log(
            "learning_rate", current_lr, prog_bar=True, on_step=False, on_epoch=True
        )

    def validation_step(self, batch, batch_idx):
        """Defines one step of validation."""
        loss, recon_loss, wasserstein_loss = self._common_step(batch, batch_idx)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_recon_loss", recon_loss, on_step=False, on_epoch=True)
        if self.use_wasserstein:
            self.log(
                "val_wasserstein_loss", wasserstein_loss, on_step=False, on_epoch=True
            )
        return loss

    def configure_optimizers(self):
        """Sets up the optimizer and learning rate scheduler."""
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
