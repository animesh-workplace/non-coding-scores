import os
import torch
import numpy as np
import torch.nn as nn
import fireducks.pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import pearsonr, wasserstein_distance


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
# Model files
# ---------------------------
models = {
    "model1": "output/model_20250910_124721.pt",
    "model2": "output/model_20250910_104113.pt",
    "model3": "output/model_20250910_101549.pt",
    "model4": "output/model_20250910_101055.pt",
    "model5": "output/model_20250910_144352.pt",
    "model6": "output/model_20250910_182929.pt",
    "model7": "output/model_20250910_191838.pt",
}


# ---------------------------
# Load test data
# ---------------------------
df = pd.read_feather("data/combined_scores.feather")
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
X_tensor = torch.tensor(X, dtype=torch.float32)
test_dataset = TensorDataset(X_tensor)
test_loader = DataLoader(test_dataset, batch_size=16384, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------
# Evaluation helper
# ---------------------------
def evaluate_model(model_path, X, X_tensor):
    model = AutoEncoder()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Run inference
    reconstructions = []
    with torch.no_grad():
        for (x,) in test_loader:
            x = x.to(device)
            x_hat = model(x)
            reconstructions.append(x_hat.cpu())
    reconstructions = torch.cat(reconstructions, dim=0).numpy()

    # Global metrics
    mse = ((X - reconstructions) ** 2).mean()
    mae = np.abs(X - reconstructions).mean()

    # Per-feature metrics
    feature_mae = np.mean(np.abs(X - reconstructions), axis=0)
    feature_mse = np.mean((X - reconstructions) ** 2, axis=0)
    feature_r2 = [r2_score(X[:, i], reconstructions[:, i]) for i in range(X.shape[1])]
    feature_corr = [
        pearsonr(X[:, i], reconstructions[:, i])[0] for i in range(X.shape[1])
    ]
    feature_wdist = [
        wasserstein_distance(X[:, i], reconstructions[:, i]) for i in range(X.shape[1])
    ]

    return mse, mae, feature_mae, feature_mse, feature_r2, feature_corr, feature_wdist


# ---------------------------
# Run evaluation for all models
# ---------------------------
results_global = []
results_features = []

for name, path in models.items():
    print(f"Evaluating {name} ...")
    mse, mae, f_mae, f_mse, f_r2, f_corr, f_wdist = evaluate_model(path, X, X_tensor)

    results_global.append([name, mse, mae])

    for i, feat in enumerate(score_cols):
        results_features.append(
            [name, feat, f_mae[i], f_mse[i], f_r2[i], f_corr[i], f_wdist[i]]
        )

# ---------------------------
# Save results
# ---------------------------
os.makedirs("output/metrics", exist_ok=True)

global_df = pd.DataFrame(results_global, columns=["model", "MSE", "MAE"])
global_df.to_csv("output/metrics/global_metrics.csv", index=False)

feature_df = pd.DataFrame(
    results_features,
    columns=["model", "feature", "MAE", "MSE", "R2", "PearsonCorr", "WasserteinDist"],
)
feature_df.to_csv("output/metrics/feature_metrics.csv", index=False)

print("✅ Saved global_metrics.csv and feature_metrics.csv")

# ---------------------------
# Heatmap of per-feature R²
# ---------------------------
pivot_r2 = feature_df.pivot(index="feature", columns="model", values="R2")
plt.figure(figsize=(10, 8))
plt.imshow(pivot_r2, aspect="auto", cmap="coolwarm", interpolation="nearest")
plt.colorbar(label="R² Score")
plt.xticks(range(len(pivot_r2.columns)), pivot_r2.columns, rotation=45)
plt.yticks(range(len(pivot_r2.index)), pivot_r2.index)
plt.title("Per-feature R² by Model")
plt.tight_layout()
plt.savefig("output/metrics/r2_heatmap.png", dpi=300)
print("✅ Saved R² heatmap")

# ---------------------------
# Heatmap of per-feature PearsonCorr
# ---------------------------
pivot_r2 = feature_df.pivot(index="feature", columns="model", values="PearsonCorr")
plt.figure(figsize=(10, 8))
plt.imshow(pivot_r2, aspect="auto", cmap="coolwarm", interpolation="nearest")
plt.colorbar(label="PearsonCorr Score")
plt.xticks(range(len(pivot_r2.columns)), pivot_r2.columns, rotation=45)
plt.yticks(range(len(pivot_r2.index)), pivot_r2.index)
plt.title("Per-feature PearsonCorr by Model")
plt.tight_layout()
plt.savefig("output/metrics/pear_heatmap.png", dpi=300)
print("✅ Saved PearsonCorr heatmap")


# ---------------------------
# Heatmap of per-feature MSE
# ---------------------------
pivot_r2 = feature_df.pivot(index="feature", columns="model", values="MSE")
plt.figure(figsize=(10, 8))
plt.imshow(pivot_r2, aspect="auto", cmap="Reds", interpolation="nearest")
plt.colorbar(label="MSE Score")
plt.xticks(range(len(pivot_r2.columns)), pivot_r2.columns, rotation=45)
plt.yticks(range(len(pivot_r2.index)), pivot_r2.index)
plt.title("Per-feature MSE by Model")
plt.tight_layout()
plt.savefig("output/metrics/mse_heatmap.png", dpi=300)
print("✅ Saved MSE heatmap")


# ---------------------------
# Heatmap of per-feature MAE
# ---------------------------
pivot_r2 = feature_df.pivot(index="feature", columns="model", values="MAE")
plt.figure(figsize=(10, 8))
plt.imshow(pivot_r2, aspect="auto", cmap="Reds", interpolation="nearest")
plt.colorbar(label="MAE Score")
plt.xticks(range(len(pivot_r2.columns)), pivot_r2.columns, rotation=45)
plt.yticks(range(len(pivot_r2.index)), pivot_r2.index)
plt.title("Per-feature MAE by Model")
plt.tight_layout()
plt.savefig("output/metrics/mae_heatmap.png", dpi=300)
print("✅ Saved MAE heatmap")


# ---------------------------
# Heatmap of per-feature WasserteinDist
# ---------------------------
pivot_r2 = feature_df.pivot(index="feature", columns="model", values="WasserteinDist")
plt.figure(figsize=(10, 8))
plt.imshow(pivot_r2, aspect="auto", cmap="Reds", interpolation="nearest")
plt.colorbar(label="WasserteinDist Score")
plt.xticks(range(len(pivot_r2.columns)), pivot_r2.columns, rotation=45)
plt.yticks(range(len(pivot_r2.index)), pivot_r2.index)
plt.title("Per-feature WasserteinDist by Model")
plt.tight_layout()
plt.savefig("output/metrics/WasserteinDist_heatmap.png", dpi=300)
print("✅ Saved WasserteinDist heatmap")
