import os
import torch
import numpy as np
import fireducks.pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance, spearmanr
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    explained_variance_score,
)


def create_reconstructions(model, val_loader, requires_mask=False):
    model.eval()
    with torch.no_grad():
        reconstructions = []
        for batch in val_loader:
            inputs = batch[0]
            if requires_mask:
                # For masked models, add zero mask to validation inputs
                zero_mask_batch = torch.zeros_like(inputs)
                inputs_with_mask = torch.cat([inputs, zero_mask_batch], dim=1)
                outputs = model(inputs_with_mask)
            else:
                outputs = model(inputs)
            reconstructions.append(outputs)

    return torch.cat(reconstructions, dim=0)


def log_cleanup(INPUT_CSV_PATH, OUTPUT_CSV_PATH):
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


def calculate_reconstruction_metrics(
    original, reconstructed, output_file, feature_names=None
):
    """
    Calculate comprehensive reconstruction metrics including per-feature metrics.

    Parameters:
    original (array-like): Original data
    reconstructed (array-like): Reconstructed data
    output_file (str): Path to save the metrics
    feature_names (list): List of feature names for per-feature metrics

    Returns:
    dict: Dictionary containing various reconstruction metrics
    """
    # Flatten arrays for 1D distance metrics
    original_flat = original.flatten()
    reconstructed_flat = reconstructed.flatten()

    # Global metrics
    metrics = {
        "MSE": mean_squared_error(original_flat, reconstructed_flat),
        "RMSE": np.sqrt(mean_squared_error(original_flat, reconstructed_flat)),
        "MAE": mean_absolute_error(original_flat, reconstructed_flat),
        "R2_Score": r2_score(original_flat, reconstructed_flat),
        "Explained_Variance": explained_variance_score(
            original_flat, reconstructed_flat
        ),
        "Wasserstein_Distance": wasserstein_distance(original_flat, reconstructed_flat),
    }

    # Per-feature metrics for multi-dimensional data
    if original.ndim > 1 and original.shape[1] > 1:
        n_features = original.shape[1]
        per_feature_metrics = {
            "MSE": [],
            "RMSE": [],
            "MAE": [],
            "R2_Score": [],
            "Explained_Variance": [],
            "Wasserstein_Distance": [],
            "Pearson_Correlation": [],
            "Spearman_Correlation": [],
        }

        for i in range(n_features):
            orig_feature = original[:, i]
            recon_feature = reconstructed[:, i]

            # Calculate all metrics for this feature
            per_feature_metrics["MSE"].append(
                mean_squared_error(orig_feature, recon_feature)
            )
            per_feature_metrics["RMSE"].append(
                np.sqrt(mean_squared_error(orig_feature, recon_feature))
            )
            per_feature_metrics["MAE"].append(
                mean_absolute_error(orig_feature, recon_feature)
            )
            per_feature_metrics["R2_Score"].append(
                r2_score(orig_feature, recon_feature)
            )
            per_feature_metrics["Explained_Variance"].append(
                explained_variance_score(orig_feature, recon_feature)
            )
            per_feature_metrics["Wasserstein_Distance"].append(
                wasserstein_distance(orig_feature, recon_feature)
            )
            per_feature_metrics["Pearson_Correlation"].append(
                np.corrcoef(orig_feature, recon_feature)[0, 1]
            )
            per_feature_metrics["Spearman_Correlation"].append(
                spearmanr(orig_feature, recon_feature)[0]
            )

        # Add summary statistics to global metrics
        for metric_name, values in per_feature_metrics.items():
            metrics[f"{metric_name}_Mean"] = np.mean(values)
            metrics[f"{metric_name}_Std"] = np.std(values)
            metrics[f"{metric_name}_Min"] = np.min(values)
            metrics[f"{metric_name}_Max"] = np.max(values)
            metrics[f"{metric_name}_Median"] = np.median(values)

        # Save detailed per-feature metrics if feature names are provided
        if feature_names is not None and len(feature_names) == n_features:
            per_feature_df = pd.DataFrame(per_feature_metrics, index=feature_names)

            # Add feature names as a column
            per_feature_df["Feature"] = feature_names
            per_feature_df = per_feature_df[
                ["Feature"] + list(per_feature_metrics.keys())
            ]

            # Save per-feature metrics to a separate file
            per_feature_file = output_file.replace(".tsv", "_per_feature.tsv")
            per_feature_df.to_csv(per_feature_file, sep="\t", index=False)
            print(f"Per-feature metrics saved to: {per_feature_file}")

        # Also save the raw per-feature arrays for further analysis
        metrics["_per_feature_arrays"] = per_feature_metrics

    # Save global metrics
    metrics_df = pd.DataFrame([metrics])

    # Remove the temporary arrays before saving
    metrics_to_save = {k: v for k, v in metrics.items() if not k.startswith("_")}
    metrics_df = pd.DataFrame([metrics_to_save])

    metrics_df.to_csv(output_file, sep="\t", index=False)
    print(f"Global metrics saved to: {output_file}")

    return metrics


def feature_importance_analysis(
    model, dataloader, feature_names, output_file, requires_mask=False
):
    """Use gradient-based feature importance with support for masked models"""
    model.eval()
    gradients = []

    for batch in dataloader:
        inputs = batch[0]

        # Handle masked models that require additional mask input
        if requires_mask:
            # Create zero mask (no corruption for importance analysis)
            zero_mask = torch.zeros_like(inputs)
            inputs_with_mask = torch.cat([inputs, zero_mask], dim=1)
            inputs = inputs_with_mask.requires_grad_(True)
        else:
            inputs = inputs.requires_grad_(True)

        outputs = model(inputs)

        # For masked models, the target is still the original input (without mask)
        target = batch[0] if requires_mask else inputs

        loss = torch.nn.MSELoss()(outputs, target)
        loss.backward()

        # For masked models, we only care about gradients of the original features (first 24)
        if requires_mask:
            gradients.append(
                inputs.grad[:, : len(feature_names)].abs().mean(dim=0).numpy()
            )
        else:
            gradients.append(inputs.grad.abs().mean(dim=0).numpy())

    avg_gradients = np.mean(gradients, axis=0)
    important_scores = dict(zip(feature_names, avg_gradients))
    sorted_importance = sorted(
        important_scores.items(), key=lambda x: x[1], reverse=True
    )
    pd.DataFrame(sorted_importance, columns=["Feature", "Importance"]).to_csv(
        output_file, sep="\t", index=False
    )


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
        correlations_sorted.to_csv(
            f"{os.path.dirname(output_path)}/per_feature_correlation.tsv", sep="\t"
        )
        print(f"Saved composite score correlation plot: {output_path}")
    else:
        plt.show()


def plot_reconstruction_metrics(metrics_files, model_names, output_path):
    """
    Create comparative visualization of reconstruction metrics across models

    Parameters:
    metrics_files: List of paths to metrics files for each model
    model_names: List of model names corresponding to the files
    output_path: Where to save the visualization
    """
    # Read and combine metrics
    all_metrics = []
    for i, file_path in enumerate(metrics_files):
        metrics_df = pd.read_csv(file_path, sep="\t")
        metrics_df["Model"] = model_names[i]
        all_metrics.append(metrics_df)

    combined_metrics = pd.concat(all_metrics, ignore_index=True)

    # Create visualization
    metrics_to_plot = [
        "MSE",
        "RMSE",
        "MAE",
        "R2_Score",
        "Explained_Variance",
        "Wasserstein_Distance",
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for i, metric in enumerate(metrics_to_plot):
        if i < len(axes):
            # Create bar plot for this metric
            axes[i].bar(model_names, combined_metrics[metric])
            axes[i].set_title(metric)
            axes[i].tick_params(axis="x", rotation=45)

            # Add value labels on bars
            for j, value in enumerate(combined_metrics[metric]):
                axes[i].text(j, value, f"{value:.4f}", ha="center", va="bottom")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    return combined_metrics


def plot_feature_importance(
    importance_files, model_names, feature_names, output_path, top_n=15
):
    """
    Create comparative visualization of feature importance across models

    Parameters:
    importance_files: List of paths to feature importance files
    model_names: List of model names corresponding to the files
    feature_names: List of all feature names
    output_path: Where to save the visualization
    top_n: Number of top features to display
    """
    # Read and combine importance data
    all_importance = {}
    for i, file_path in enumerate(importance_files):
        importance_df = pd.read_csv(file_path, sep="\t")
        all_importance[model_names[i]] = dict(
            zip(importance_df["Feature"], importance_df["Importance"])
        )

    # Create a matrix of importance values
    importance_matrix = np.zeros((len(feature_names), len(model_names)))
    for j, model in enumerate(model_names):
        for i, feature in enumerate(feature_names):
            importance_matrix[i, j] = all_importance[model].get(feature, 0)

    # Get top features across all models
    avg_importance = np.mean(importance_matrix, axis=1)
    top_indices = np.argsort(avg_importance)[-top_n:]
    top_features = [feature_names[i] for i in top_indices]

    # Create heatmap
    plt.figure(figsize=(12, 10))
    im = plt.imshow(importance_matrix[top_indices, :], cmap="viridis", aspect="auto")

    # Customize plot
    plt.yticks(range(len(top_features)), top_features)
    plt.xticks(range(len(model_names)), model_names, rotation=45)
    plt.xlabel("Models")
    plt.ylabel("Features")
    plt.title("Feature Importance Across Models")

    # Add colorbar
    cbar = plt.colorbar(im)
    cbar.set_label("Importance Score")

    # Add value annotations
    for i in range(len(top_features)):
        for j in range(len(model_names)):
            plt.text(
                j,
                i,
                f"{importance_matrix[top_indices[i], j]:.3f}",
                ha="center",
                va="center",
                color="white"
                if importance_matrix[top_indices[i], j] > 0.5
                else "black",
            )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    return importance_matrix


def plot_per_feature_reconstruction(
    metrics_files, model_names, feature_names, output_path, metric_name="R2_Score"
):
    """
    Create visualization of per-feature reconstruction metrics across models

    Parameters:
    metrics_files: List of paths to per-feature metrics files
    model_names: List of model names corresponding to the files
    feature_names: List of all feature names
    output_path: Where to save the visualization
    metric_name: Which metric to visualize (e.g., 'R2_Score', 'MSE')
    """
    # Read and combine data
    all_metrics = {}
    for i, file_path in enumerate(metrics_files):
        metrics_df = pd.read_csv(file_path, sep="\t")
        all_metrics[model_names[i]] = dict(
            zip(metrics_df["Feature"], metrics_df[metric_name])
        )

    # Create a matrix of metric values
    metric_matrix = np.zeros((len(feature_names), len(model_names)))
    for j, model in enumerate(model_names):
        for i, feature in enumerate(feature_names):
            metric_matrix[i, j] = all_metrics[model].get(feature, 0)

    # Create heatmap
    plt.figure(figsize=(12, 10))
    im = plt.imshow(metric_matrix, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)

    # Customize plot
    plt.yticks(range(len(feature_names)), feature_names)
    plt.xticks(range(len(model_names)), model_names, rotation=45)
    plt.xlabel("Models")
    plt.ylabel("Features")
    plt.title(f"Per-Feature {metric_name} Across Models")

    # Add colorbar
    cbar = plt.colorbar(im)
    cbar.set_label(metric_name)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    return metric_matrix
