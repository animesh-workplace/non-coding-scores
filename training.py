import os
import torch
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
X = df[score_cols].apply(pd.to_numeric, errors="coerce").fillna(0).values
print("Score only done")

# Convert to Torch tensor
X_tensor = torch.tensor(X, dtype=torch.float32)
print("Tensor")

# Dataset & DataLoader
num_workers = os.cpu_count() // 4
dataset = TensorDataset(X_tensor)
print("Tensor dataset")
dataloader = DataLoader(dataset, batch_size=8192, shuffle=True, num_workers=num_workers)
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

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out

    def training_step(self, batch, batch_idx):
        (x,) = batch
        x_hat = self(x)
        # loss = nn.MSELoss()(x_hat, x)
        loss = nn.L1Loss()(x_hat, x)
        self.epoch_loss.append(loss.detach())
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def on_train_epoch_end(self):
        # check if list is not empty
        if len(self.epoch_loss) > 0:
            avg_loss = torch.stack(self.epoch_loss).mean().item()
            self.train_losses.append(avg_loss)
        else:
            self.train_losses.append(float("nan"))
        # reset for next epoch
        self.epoch_loss = []

    def configure_optimizers(self):
        # return torch.optim.Adam(self.parameters(), lr=1e-3, weight_decay=1e-5)
        # return torch.optim.SGD(self.parameters(), lr=0.01, momentum=0.9)
        return torch.optim.NAdam(self.parameters(), lr=1e-3, weight_decay=1e-5)


# ---------------------------
# Training
# ---------------------------
model = AutoEncoder()
early_stopping = EarlyStopping(
    monitor="train_loss", patience=5, mode="min", verbose=True
)
trainer = pl.Trainer(
    max_epochs=50, accelerator="auto", log_every_n_steps=1, callbacks=[early_stopping]
)
trainer.fit(model, dataloader)

# ---------------------------
# Extract Composite Scores
# ---------------------------
model.eval()
with torch.no_grad():
    composite_scores = model.encoder(X_tensor).squeeze().numpy()

# Attach to metadata
result = df[["chromosome", "position", "ref", "alt"]].copy()
result["composite_score"] = composite_scores

# Save
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
os.makedirs("output", exist_ok=True)
result.to_feather(f"output/composite_scores_{timestamp}.feather")
print("Saved composite_scores")

plt.figure(figsize=(8, 5))
plt.plot(model.train_losses, marker="o", label="Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Training Loss (L1Loss)")
plt.title("Training Loss vs Epoch")
plt.grid(True)
plt.legend()

# Save the plot as a PNG file
plt.savefig(f"output/training_loss_{timestamp}.png", dpi=300, bbox_inches="tight")
print("Saved training loss plot")
