import os
import torch
import numpy as np
import fireducks.pandas as pd
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------
# Autoencoder Model (same as training)
# ---------------------------
class AutoEncoder(nn.Module):
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

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out


# ---------------------------
# Load model
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoEncoder()
model.load_state_dict(
    torch.load("output/model_20250910_101055.pt", map_location=device)
)
model.to(device)
model.eval()
print("✅ Model loaded successfully")


# ---------------------------
# Load test data
# ---------------------------
df = pd.read_feather("data/combined_scores.feather")
score_cols = [
    # "chr",
    # "pos",
    # "ref",
    # "alt",
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
    # "fathmm_xf_coding",
    "fathmm_mkl_coding",
    "fathmm_xf_noncoding",
    "fathmm_mkl_noncoding",
    "conservation_30p",
    "conservation_100v",
    # "na_count",
    # "gene"
]

X = df[score_cols].apply(pd.to_numeric, errors="coerce").fillna(0).values
X_tensor = torch.tensor(X, dtype=torch.float32)
test_dataset = TensorDataset(X_tensor)
test_loader = DataLoader(test_dataset, batch_size=16384, shuffle=False)
print("✅ Test data loaded")


# ---------------------------
# Baseline: predict mean of each feature
# ---------------------------
feature_means = X.mean(axis=0, keepdims=True)  # (1, 24)
baseline_preds = np.repeat(feature_means, X.shape[0], axis=0)

baseline_mse = ((X - baseline_preds) ** 2).mean()
baseline_mae = np.abs(X - baseline_preds).mean()

print(f"Baseline MSE (predicting feature means): {baseline_mse:.6f}")
print(f"Baseline MAE (predicting feature means): {baseline_mae:.6f}")

# ---------------------------
# Run inference
# ---------------------------
reconstructions = []
with torch.no_grad():
    for batch in test_loader:
        (x,) = batch
        x = x.to(device)
        x_hat = model(x)
        reconstructions.append(x_hat.cpu())

reconstructions = torch.cat(reconstructions, dim=0).numpy()
print("✅ Inference complete")


# ---------------------------
# Evaluate reconstruction error
# ---------------------------
mse = ((X - reconstructions) ** 2).mean()
mae = np.abs(X - reconstructions).mean()
print(f"Mean Squared Error (MSE): {mse:.6f}")
print(f"Mean Absolute Error (MAE): {mae:.6f}")


# ---------------------------
# Extract composite scores (bottleneck representation)
# ---------------------------
with torch.no_grad():
    composite_scores = model.encoder(X_tensor.to(device)).squeeze().cpu().numpy()

result = df[["chr", "pos", "ref", "alt"]].copy()
result["composite_score"] = composite_scores

os.makedirs("output", exist_ok=True)
result.to_feather("output/composite_scores_test.feather")
print("✅ Composite scores saved")


# ---------------------------
# Visualization
# ---------------------------
n_samples = 5
fig, axes = plt.subplots(n_samples, 2, figsize=(10, 10))

for i in range(n_samples):
    axes[i, 0].plot(X[i], label="Original", color="blue")
    axes[i, 1].plot(reconstructions[i], label="Reconstructed", color="orange")
    axes[i, 0].legend()
    axes[i, 1].legend()

plt.tight_layout()
plt.savefig("output/reconstruction_examples.png", dpi=300, bbox_inches="tight")
print("✅ Plots saved: output/reconstruction_examples.png")
