import os
import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import lightning as pl
from datetime import datetime
import fireducks.pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from lightning.pytorch.callbacks import EarlyStopping
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------
# Load your data
# ---------------------------
df = pd.read_feather("data/combined_scores.feather")
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
X_train, X_val = train_test_split(X_tensor, test_size=0.2, random_state=42)
print("Tensor")

# Dataset & DataLoader
num_workers = os.cpu_count() // 4
train_dataset = TensorDataset(X_train)
val_dataset = TensorDataset(X_val)

print("Tensor dataset")
train_loader = DataLoader(
    train_dataset,
    batch_size=16384,
    shuffle=True,
    pin_memory=False,
    num_workers=num_workers,
    persistent_workers=True,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=16384,
    shuffle=False,
    pin_memory=False,
    num_workers=num_workers,
    persistent_workers=True,
)

print("DataLoader")


# ---------------------------
# Denoising Autoencoder Model
# ---------------------------
class DenoisingAutoEncoder(pl.LightningModule):
    def __init__(self, corruption_level=0.1, corruption_value=99999):
        super().__init__()
        # MODIFICATION: Added corruption_level and corruption_value to __init__
        self.corruption_level = corruption_level
        self.corruption_value = corruption_value

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
        self.val_losses = []  # Track validation losses
        self.train_losses = []
        self.learning_rates = []  # Track learning rates
        self.lr_change_epochs = []  # Track when LR changes

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out

    def training_step(self, batch, batch_idx):
        (x,) = batch

        # Initialize epoch_loss if it doesn't exist
        if not hasattr(self, "epoch_loss"):
            self.epoch_loss = []

        # MODIFICATION: Denoising logic with custom corruption value
        # 1. Create a copy of the input to avoid modifying the original data
        x_noisy = x.clone()

        # 2. Create a random mask to select which values to corrupt
        corruption_mask = torch.rand_like(x) < self.corruption_level

        # 3. Apply corruption by setting masked values to the specified corruption_value
        x_noisy[corruption_mask] = self.corruption_value

        # 4. Pass the corrupted ("noisy") input to the model
        x_hat = self(x_noisy)

        # 5. Calculate loss against the ORIGINAL, clean input to train the model to reconstruct it
        loss = nn.L1Loss()(x_hat, x)

        self.epoch_loss.append(loss.detach())
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # Validation remains the same: evaluate the model's ability to reconstruct clean data
        (x,) = batch
        x_hat = self(x)
        loss = nn.L1Loss()(x_hat, x)
        self.log(
            "val_loss", loss, prog_bar=True, on_step=False, on_epoch=True
        )  # Log validation loss
        return loss

    def on_validation_epoch_end(self):
        # Store validation loss
        if hasattr(self, "trainer") and hasattr(self.trainer, "callback_metrics"):
            val_loss = self.trainer.callback_metrics.get("val_loss")
            if val_loss is not None:
                self.val_losses.append(val_loss.item())

    def on_train_epoch_end(self):
        # Store average loss for the epoch
        if self.epoch_loss:
            avg_loss = torch.stack(self.epoch_loss).mean().item()
            self.train_losses.append(avg_loss)
        else:
            self.train_losses.append(float("nan"))

        # Store current learning rate - FIXED: Access first optimizer in list
        if (
            hasattr(self, "trainer")
            and hasattr(self.trainer, "optimizers")
            and self.trainer.optimizers
        ):
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
                "monitor": "val_loss",
            },
        }


# ---------------------------
# Training
# ---------------------------
# MODIFICATION: Instantiate the DenoisingAutoEncoder with the custom corruption value
model = DenoisingAutoEncoder(corruption_level=0.1, corruption_value=99999)
early_stopping = EarlyStopping(
    monitor="val_loss", patience=10, mode="min", verbose=True
)
trainer = pl.Trainer(
    max_epochs=500, accelerator="auto", log_every_n_steps=1, callbacks=[early_stopping]
)
trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

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


# Create comprehensive figure with three subplots
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12))

# Plot 1: Training and Validation Loss
ax1.plot(model.train_losses, marker="o", label="Training Loss", color="blue", alpha=0.8)
if model.val_losses:
    ax1.plot(
        model.val_losses, marker="s", label="Validation Loss", color="red", alpha=0.8
    )
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss (L1Loss)")
ax1.set_title("Training and Validation Loss vs Epoch")
ax1.grid(True, alpha=0.3)
ax1.legend()

# Mark learning rate change points on loss plot
for epoch in model.lr_change_epochs:
    if epoch < len(model.train_losses):
        ax1.axvline(x=epoch, color="green", linestyle="--", alpha=0.7)
        ax1.annotate(
            "LR↓",
            xy=(
                epoch,
                max(
                    model.train_losses[epoch],
                    model.val_losses[epoch]
                    if model.val_losses and epoch < len(model.val_losses)
                    else model.train_losses[epoch],
                ),
            ),
            xytext=(5, 5),
            textcoords="offset points",
            color="green",
            fontweight="bold",
        )

# Plot 2: Learning Rate
ax2.plot(model.learning_rates, marker="s", label="Learning Rate", color="green")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Learning Rate")
ax2.set_title("Learning Rate vs Epoch")
ax2.grid(True, alpha=0.3)
ax2.set_yscale("log")  # Log scale for better visualization of LR changes
ax2.legend()

# Mark learning rate change points on LR plot
for epoch in model.lr_change_epochs:
    if epoch < len(model.learning_rates):
        ax2.axvline(x=epoch, color="red", linestyle="--", alpha=0.7)
        ax2.annotate(
            "LR↓",
            xy=(epoch, model.learning_rates[epoch]),
            xytext=(5, 5),
            textcoords="offset points",
            color="red",
            fontweight="bold",
        )

# Plot 3: Loss Ratio (Train/Val) to show overfitting
if model.val_losses and len(model.val_losses) == len(model.train_losses):
    loss_ratio = [
        train_loss / val_loss if val_loss > 0 else 0
        for train_loss, val_loss in zip(model.train_losses, model.val_losses)
    ]
    ax3.plot(loss_ratio, marker="^", label="Train/Val Loss Ratio", color="purple")
    ax3.axhline(
        y=1.0, color="gray", linestyle="-", alpha=0.5, label="Ideal Ratio (1.0)"
    )
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Train/Val Loss Ratio")
    ax3.set_title("Training to Validation Loss Ratio (Lower = Better Generalization)")
    ax3.grid(True, alpha=0.3)
    ax3.legend()

plt.tight_layout()

# Save the comprehensive plot as a PNG file
plt.savefig(
    f"output/comprehensive_training_metrics_{timestamp}.png",
    dpi=300,
    bbox_inches="tight",
)
print("Saved comprehensive training metrics plot")


# Print final metrics
print(f"\nFinal Training Loss: {model.train_losses[-1]:.6f}")
if model.val_losses:
    print(f"Final Validation Loss: {model.val_losses[-1]:.6f}")
    print(f"Final Train/Val Ratio: {model.train_losses[-1] / model.val_losses[-1]:.3f}")
print(f"Final Learning Rate: {model.learning_rates[-1]:.2e}")
