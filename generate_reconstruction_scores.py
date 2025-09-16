import torch
import numpy as np
from tqdm import tqdm
import fireducks.pandas as pd
import matplotlib.pyplot as plt
from utils import create_reconstructions
from autoencoders.base_ae import AutoEncoder
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import ks_2samp, wasserstein_distance
from autoencoders.denoising_ae import DenoisingAutoEncoder
from autoencoders.orthogonal_ae import OrthogonalAutoEncoder
from autoencoders.binary_mask_dae import MaskedDenoisingAutoEncoder

BATCH_SIZE = 16384
CHUNK_SIZE = 100000
NUM_WORKERS = 4
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


def compare_score_distributions(
    name,
    df_orig,
    df_recon,
    score_cols,
    label_orig="Original",
    label_recon="Reconstructed",
):
    results = {}

    for col in score_cols:
        orig_values = df_orig[col].dropna()
        recon_values = df_recon[col].dropna()

        # KS test
        ks_stat, ks_p = ks_2samp(orig_values, recon_values)
        # Wasserstein distance
        wd = wasserstein_distance(orig_values, recon_values)
        results[col] = {"KS_stat": ks_stat, "KS_p": ks_p, "Wasserstein": wd}

        # Histogram comparison
        plt.figure(figsize=(8, 4))
        plt.hist(orig_values, bins=100, alpha=0.5, label=label_orig, density=True)
        plt.hist(recon_values, bins=100, alpha=0.5, label=label_recon, density=True)
        plt.title(f"Distribution of {col}")
        plt.xlabel("Score")
        plt.ylabel("Density")
        plt.legend()
        plt.tight_layout()
        hist_path = f"{name}_{col}_hist.png"
        plt.savefig(hist_path, dpi=300)
        plt.close()

        # QQ-plot: original vs reconstructed quantiles
        plt.figure(figsize=(5, 5))
        q_orig = pd.Series(orig_values).quantile(q=np.linspace(0, 1, 1000))
        q_recon = pd.Series(recon_values).quantile(q=np.linspace(0, 1, 1000))
        plt.plot(q_orig, q_recon, "o", alpha=0.5)
        max_val = max(q_orig.max(), q_recon.max())
        min_val = min(q_orig.min(), q_recon.min())
        plt.plot([min_val, max_val], [min_val, max_val], "r--")  # diagonal reference
        plt.title(f"QQ-plot of {col} ({label_orig} vs {label_recon})")
        plt.xlabel(f"{label_orig} Quantiles")
        plt.ylabel(f"{label_recon} Quantiles")
        plt.tight_layout()
        qq_path = f"{name}_{col}_qq.png"
        plt.savefig(qq_path, dpi=300)
        plt.close()

    # Convert results to DataFrame
    df_results = pd.DataFrame(results).T
    # Barplot for Wasserstein distance (to quickly see worst scores)
    plt.figure(figsize=(10, 5))
    df_results["Wasserstein"].sort_values(ascending=False).plot(
        kind="bar", color="steelblue"
    )
    plt.title(f"Wasserstein Distance Across Scores ({label_orig} vs {label_recon})")
    plt.ylabel("Wasserstein Distance")
    plt.tight_layout()
    barplot_path = f"{name}_wasserstein_bar.png"
    plt.savefig(barplot_path, dpi=300)
    plt.close()

    return df_results


# ========== NEW FUNCTION ADDED HERE ==========
def plot_residuals_grid(name, df_orig, df_recon, score_cols):
    """
    Generates and saves a single grid image of residual plots for all scores.
    """
    # Create a 4x6 grid of subplots for the 24 scores
    fig, axes = plt.subplots(4, 6, figsize=(24, 16))
    axes = axes.flatten()  # Flatten the 2D array of axes for easy iteration

    for i, col in enumerate(score_cols):
        ax = axes[i]
        print(df_orig[col], df_recon[col], len(df_recon[col]), len(df_orig[col]))

        # Calculate residuals for the current column
        residuals = df_orig[col].values - df_recon[col].values
        print(residuals, len(residuals))

        # Create scatter plot of original values vs. residuals
        ax.scatter(df_orig[col], residuals, alpha=0.1, s=5)

        # Add a horizontal line at y=0 for reference
        ax.axhline(y=0, color="r", linestyle="--")

        # Set titles and labels for each subplot
        ax.set_title(col, fontsize=12)
        ax.set_xlabel("Original Score", fontsize=10)
        ax.set_ylabel("Residual", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.6)

    # Add a main title to the entire figure
    fig.suptitle(f"Residual Plots for {name}", fontsize=20)

    # Adjust layout to prevent plots from overlapping
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save the combined plot
    residuals_path = f"{name}_residuals_grid.png"
    plt.savefig(residuals_path, dpi=300)
    plt.close()


# ============================================

for model_name in ["base", "denoising", "masked_denoising", "orthogonal"]:
    masking = model_name in ["orthogonal", "masked_denoising"]
    model_class = {
        "base": AutoEncoder,
        "denoising": DenoisingAutoEncoder,
        "orthogonal": OrthogonalAutoEncoder,
        "masked_denoising": MaskedDenoisingAutoEncoder,
    }
    sample_size = [1, 10, 25]
    for size in sample_size:
        df_orig = pd.read_feather(f"data/sampled_dataset_{size}M.feather")
        df_orig.reset_index(inplace=True, drop=True)
        X_chunks = []
        for i in tqdm(range(0, len(df_orig), CHUNK_SIZE)):
            chunk = df_orig.iloc[i : i + CHUNK_SIZE][score_cols]
            X_chunk = chunk.apply(pd.to_numeric, errors="coerce").fillna(0).values
            X_chunks.append(X_chunk)
        X = np.vstack(X_chunks)
        X_tensor = torch.tensor(X, dtype=torch.float32)
        dataset = TensorDataset(X_tensor)
        loader = DataLoader(
            dataset,
            shuffle=True,
            pin_memory=False,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
            persistent_workers=True,
        )
        model = model_class[model_name]()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.load_state_dict(
            torch.load(
                f"output/DL4_Training/training_{size}m/{model_name}/model.pt",
                map_location=device,
            )
        )
        model.to(device)
        model.eval()

        df_recon = pd.DataFrame(
            create_reconstructions(model, loader, masking).numpy(), columns=score_cols
        )
        # print(f"Generating QQ and histogram plots for {model_name}_{size}M...")
        # compare_score_distributions(
        #     f"{model_name}_{size}M",
        #     df_orig,
        #     df_recon,
        #     score_cols,
        #     label_orig="Original",
        #     label_recon=f"Reconstructed_{model_name}_{size}M",
        # )
        print(f"Generating residual plots for {model_name}_{size}M...")
        plot_residuals_grid(
            f"{model_name}_{size}M", df_orig.reset_index(), df_recon, score_cols
        )
