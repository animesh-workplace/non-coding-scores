import os
import torch
import numpy as np
import seaborn as sns
import fireducks.pandas as pd
import matplotlib.pyplot as plt


def log_cleanup(INPUT_CSV_PATH, OUTPUT_CSV_PATH):
    output_dir = os.path.dirname(OUTPUT_CSV_PATH)
    os.makedirs(output_dir, exist_ok=True)
    try:
        df = pd.read_csv(INPUT_CSV_PATH)
    except FileNotFoundError:
        print(f"Error: Could not find the file at '{INPUT_CSV_PATH}'")
        exit()

    df["epoch"].ffill(inplace=True)
    df["learning_rate"].ffill(inplace=True)

    df_cleaned = df.groupby("epoch").first()
    df_cleaned["step"] = df.groupby("epoch")["step"].last()
    df_cleaned.to_csv(OUTPUT_CSV_PATH, sep="\t", index=False)
    print(f"\nMetrics have been saved to: {OUTPUT_CSV_PATH}")


# def save_model(model, data, X_tensor, output_path):
#     model.eval()
#     with torch.no_grad():
#         composite_scores = model.encoder(X_tensor).squeeze().numpy()

#     result = data[["chr", "pos", "ref", "alt"]].copy()
#     result["composite_score"] = composite_scores
#     result.to_feather(f"{output_path}/composite_scores.feather")
#     torch.save(model.state_dict(), f"{output_path}/model.pt")
#     print(f"Saved composite_scores & model params: {output_path}")


def save_model(model, data, X_tensor, output_path):
    model.eval()
    device = next(model.parameters()).device
    X_tensor = X_tensor.to(device)
    with torch.no_grad():
        latent_representation = model.encoder(X_tensor).cpu().numpy()
    result = data[["chr", "pos", "ref", "alt"]].copy()
    # For models with multiple latent dimensions
    if latent_representation.shape[1] > 1:
        # Create columns for each latent dimension
        for i in range(latent_representation.shape[1]):
            result[f"latent_dim_{i}"] = latent_representation[:, i]
        # Optionally, you can create a composite score as the norm of the latent vector
        result["composite_score"] = np.linalg.norm(latent_representation, axis=1)
    else:
        # For models with a single latent dimension
        result["composite_score"] = latent_representation.squeeze()
    result.to_feather(f"{output_path}/composite_scores.feather")
    torch.save(model.state_dict(), f"{output_path}/model.pt")
    print(f"Saved composite_scores & model params: {output_path}")


def plot_training_metrics(data_size, csv_path, output_path=None):
    """
    Create comprehensive training visualization from CSVLogger output

    Parameters:
    csv_path (str): Path to the metrics.csv file
    output_path (str, optional): Path to save the plot. If None, shows the plot
    """
    # Read the CSV file
    df = pd.read_csv(csv_path, sep="\t")

    # Create epoch numbers (assuming each row is an epoch)
    df["epoch"] = range(1, len(df) + 1)

    # Find learning rate change points
    lr_changes = []
    for i in range(1, len(df)):
        if df["learning_rate"].iloc[i] != df["learning_rate"].iloc[i - 1]:
            lr_changes.append(df["epoch"].iloc[i])

    # Create comprehensive figure with three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12))

    # Plot 1: Training and Validation Loss
    ax1.plot(
        df["epoch"],
        df["train_loss"],
        marker="o",
        label="Training Loss",
        color="blue",
        alpha=0.8,
    )
    ax1.plot(
        df["epoch"],
        df["val_loss"],
        marker="s",
        label="Validation Loss",
        color="red",
        alpha=0.8,
    )
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss (L1Loss)")
    ax1.set_title(f"Training and Validation Loss vs Epoch ({data_size:.1f}M)")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Mark learning rate change points on loss plot
    for epoch in lr_changes:
        ax1.axvline(x=epoch, color="green", linestyle="--", alpha=0.7)
        ax1.annotate(
            "LR↓",
            xy=(
                epoch,
                max(df["train_loss"].iloc[epoch - 1], df["val_loss"].iloc[epoch - 1]),
            ),
            xytext=(5, 5),
            textcoords="offset points",
            color="green",
            fontweight="bold",
        )

    # Plot 2: Learning Rate
    ax2.plot(
        df["epoch"],
        df["learning_rate"],
        marker="s",
        label="Learning Rate",
        color="green",
    )
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Learning Rate")
    ax2.set_title(f"Learning Rate vs Epoch ({data_size:.1f}M)")
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale("log")  # Log scale for better visualization of LR changes
    ax2.legend()

    # Mark learning rate change points on LR plot
    for epoch in lr_changes:
        ax2.axvline(x=epoch, color="red", linestyle="--", alpha=0.7)
        ax2.annotate(
            "LR↓",
            xy=(epoch, df["learning_rate"].iloc[epoch - 1]),
            xytext=(5, 5),
            textcoords="offset points",
            color="red",
            fontweight="bold",
        )

    # Plot 3: Loss Ratio (Train/Val) to show overfitting
    loss_ratio = df["train_loss"] / df["val_loss"]
    ax3.plot(
        df["epoch"],
        loss_ratio,
        marker="^",
        label="Train/Val Loss Ratio",
        color="purple",
    )
    ax3.axhline(
        y=1.0, color="gray", linestyle="-", alpha=0.5, label="Ideal Ratio (1.0)"
    )

    # Mark learning rate change points on loss ratio plot
    for epoch in lr_changes:
        ax3.axvline(x=epoch, color="orange", linestyle="--", alpha=0.7)
        ax3.annotate(
            "LR↓",
            xy=(epoch, loss_ratio.iloc[epoch - 1]),
            xytext=(5, 5),
            textcoords="offset points",
            color="orange",
            fontweight="bold",
        )

    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Train/Val Loss Ratio")
    ax3.set_title(
        f"Training to Validation Loss Ratio (Lower = Better Generalization) ({data_size:.1f}M)"
    )
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    plt.tight_layout()

    # Save or show the plot
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved comprehensive training metrics plot: {output_path}")
    else:
        plt.show()


def plot_correlation_comparison(
    data_size, score_cols, df_original, composite_scores_file, output_path=None
):
    df_composite = pd.read_feather(composite_scores_file)
    df_merged = pd.merge(df_composite, df_original, on=["chr", "pos", "ref", "alt"])
    correlations = (
        df_merged[score_cols + ["composite_score"]]
        .corr()["composite_score"]
        .drop("composite_score")
    )
    correlations_sorted = correlations.sort_values(ascending=False)

    plt.figure(figsize=(12, 10))

    # Create a bar plot without the deprecated palette parameter
    plt.barh(
        range(len(correlations_sorted.index)),
        correlations_sorted.values,
        color=plt.cm.viridis(np.linspace(0, 1, len(correlations_sorted))),
    )

    # Set y-axis labels to feature names
    plt.yticks(range(len(correlations_sorted.index)), correlations_sorted.index)

    plt.title(
        f"Correlation of Original Features with the Composite Score ({data_size:.1f}M)",
        fontsize=16,
    )
    plt.xlabel("Pearson Correlation Coefficient", fontsize=12)
    plt.ylabel("Original Features", fontsize=12)
    plt.grid(axis="x", linestyle="--", alpha=0.7)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved composite score correlation plot: {output_path}")
    else:
        plt.show()
