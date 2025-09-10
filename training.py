import os
import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import lightning as pl
from datetime import datetime
import fireducks.pandas as pd
import matplotlib.pyplot as plt
from lightning.pytorch.callbacks import EarlyStopping
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------
# Load your data
# ---------------------------
df = pd.read_feather("data/sampled_dataset_25M.feather")
print("Data Loaded")

# Only keep score columns (24 features)
score_cols = [
    "gpn",
    "cadd",
    "remm",
    "dann",
    "fire",
    "gerp",
    "cscape",
    "gwrvis",
    "jarvis",
    "funseq2",
    "linsight",
    "repliseq_g2",
    "repliseq_s1",
    "repliseq_s2",
    "repliseq_s3",
    "repliseq_s4",
    "repliseq_g1b",
    "macie_conserved",
    "macie_regulatory",
    "fathmm_mkl_coding",
    "fathmm_xf_noncoding",
    "fathmm_mkl_noncoding",
    "conservation_30p",
    "conservation_100v",
]

chunk_size = 100000  # Adjust based on your RAM
X_chunks = []

for i in tqdm(range(0, len(df), chunk_size)):
    chunk = df.iloc[i : i + chunk_size][score_cols]
    X_chunk = chunk.apply(pd.to_numeric, errors="coerce").fillna(0).values
    X_chunks.append(X_chunk)

X = np.vstack(X_chunks)
print("Score only done")

# Convert to Torch tensor
X_tensor = torch.tensor(X, dtype=torch.float32)
print("Tensor")

# Dataset & DataLoader
num_workers = os.cpu_count() // 4
dataset = TensorDataset(X_tensor)
print("Tensor dataset")
dataloader = DataLoader(
    dataset,
    shuffle=True,
    pin_memory=False,
    batch_size=16384,
    num_workers=num_workers,
    persistent_workers=True,
)
print("DataLoader")


# ---------------------------
# Autoencoder Model
# ---------------------------
class AutoEncoder(pl.LightningModule):
    def __init__(self):
        super().__init__()
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
        # store loss history
        self.epoch_loss = []
        self.train_losses = []
        self.learning_rates = []  # Track learning rates
        self.lr_change_epochs = []  # Track when LR changes

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out

    def training_step(self, batch, batch_idx):
        (x,) = batch
        x_hat = self(x)
        loss = nn.L1Loss()(x_hat, x)
        self.epoch_loss.append(loss.detach())
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def on_train_epoch_end(self):
        # Store average loss for the epoch
        if len(self.epoch_loss) > 0:
            avg_loss = torch.stack(self.epoch_loss).mean().item()
            self.train_losses.append(avg_loss)
        else:
            self.train_losses.append(float("nan"))

        # Store current learning rate
        current_lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.learning_rates.append(current_lr)

        # Check if learning rate changed
        if (
            len(self.learning_rates) > 1
            and self.learning_rates[-1] != self.learning_rates[-2]
        ):
            self.lr_change_epochs.append(self.current_epoch)
            print(
                f"Epoch {self.current_epoch}: Learning rate changed to {current_lr:.2e}"
            )

        # reset for next epoch
        self.epoch_loss = []

    def configure_optimizers(self):
        optimizer = torch.optim.NAdam(self.parameters(), lr=1e-3, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            patience=2,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "train_loss",
            },
        }


# ---------------------------
# Training
# ---------------------------
model = AutoEncoder()
early_stopping = EarlyStopping(
    monitor="train_loss", patience=5, mode="min", verbose=True
)
trainer = pl.Trainer(
    max_epochs=500, accelerator="auto", log_every_n_steps=1, callbacks=[early_stopping]
)
trainer.fit(model, dataloader)

# Print summary of learning rate changes
print("\nLearning Rate Change Summary:")
for epoch in model.lr_change_epochs:
    print(f"Epoch {epoch}: LR changed to {model.learning_rates[epoch]:.2e}")

# ---------------------------
# Extract Composite Scores
# ---------------------------
model.eval()
with torch.no_grad():
    composite_scores = model.encoder(X_tensor).squeeze().numpy()

# Attach to metadata
result = df[["chr", "pos", "ref", "alt"]].copy()
result["composite_score"] = composite_scores

# Save
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
os.makedirs("output", exist_ok=True)
result.to_feather(f"output/composite_scores_{timestamp}.feather")
torch.save(model.state_dict(), f"output/model_{timestamp}.pt")
print("Saved composite_scores & model params")

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# Plot 1: Training Loss
ax1.plot(model.train_losses, marker="o", label="Training Loss", color="blue")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Training Loss (L1Loss)")
ax1.set_title("Training Loss vs Epoch")
ax1.grid(True)
ax1.legend()

# Mark learning rate change points on loss plot
for epoch in model.lr_change_epochs:
    if epoch < len(model.train_losses):
        ax1.axvline(
            x=epoch,
            color="red",
            linestyle="--",
            alpha=0.7,
            label="LR Change" if epoch == model.lr_change_epochs[0] else "",
        )
        ax1.annotate(
            f"LR↓",
            xy=(epoch, model.train_losses[epoch]),
            xytext=(5, 5),
            textcoords="offset points",
            color="red",
        )

# Plot 2: Learning Rate
ax2.plot(model.learning_rates, marker="s", label="Learning Rate", color="green")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Learning Rate")
ax2.set_title("Learning Rate vs Epoch")
ax2.grid(True)
ax2.set_yscale("log")  # Log scale for better visualization of LR changes
ax2.legend()

# Mark learning rate change points on LR plot
for epoch in model.lr_change_epochs:
    if epoch < len(model.learning_rates):
        ax2.axvline(x=epoch, color="red", linestyle="--", alpha=0.7)
        ax2.annotate(
            f"LR↓",
            xy=(epoch, model.learning_rates[epoch]),
            xytext=(5, 5),
            textcoords="offset points",
            color="red",
        )

plt.tight_layout()

# Save the plot as a PNG file
plt.savefig(f"output/training_loss_lr_{timestamp}.png", dpi=300, bbox_inches="tight")
print("Saved training loss and learning rate plot")

# Also save individual plots
plt.figure(figsize=(8, 5))
plt.plot(model.train_losses, marker="o", label="Training Loss", color="blue")
for epoch in model.lr_change_epochs:
    if epoch < len(model.train_losses):
        plt.axvline(x=epoch, color="red", linestyle="--", alpha=0.7)
        plt.annotate(
            f"LR Change\nEpoch {epoch}",
            xy=(epoch, model.train_losses[epoch]),
            xytext=(10, 10),
            textcoords="offset points",
            color="red",
            arrowprops=dict(arrowstyle="->", color="red"),
        )
plt.xlabel("Epoch")
plt.ylabel("Training Loss (L1Loss)")
plt.title("Training Loss vs Epoch with LR Changes")
plt.grid(True)
plt.legend()
plt.savefig(
    f"output/training_loss_with_lr_changes_{timestamp}.png",
    dpi=300,
    bbox_inches="tight",
)

print("All plots saved")
