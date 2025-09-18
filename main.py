import os
import torch
import numpy as np
from tqdm import tqdm
import lightning as pl
from datetime import datetime
import fireducks.pandas as pd
from autoencoders.base_ae import AutoEncoder
from autoencoders.m_base import BigAutoEncoder
from lightning.pytorch.loggers import CSVLogger
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from autoencoders.weighted_ae import WeightedAutoEncoder
from autoencoders.denoising_ae import DenoisingAutoEncoder
from autoencoders.orthogonal_ae import OrthogonalAutoEncoder
from autoencoders.wassertein_ae import WassersteinAutoEncoder
from autoencoders.binary_mask_dae import MaskedDenoisingAutoEncoder
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from utils import (
    save_model,
    log_cleanup,
    plot_training_metrics,
    create_reconstructions,
    plot_feature_importance,
    feature_importance_analysis,
    plot_reconstruction_metrics,
    plot_correlation_comparison,
    plot_per_feature_reconstruction,
    calculate_reconstruction_metrics,
)

# --- Configuration ---
BATCH_SIZE = 16384
CHUNK_SIZE = 100000
WEIGHT_DECAY = 1e-5
LEARNING_RATE = 1e-3
CORRUPTION_LEVEL = 0.15  # Corrupt 15% of the input values
CORRUPTION_VALUE = 99999
MAX_TRAINING_EPOCHS = 500
EARLY_STOPPING_PATIENCE = 20
NUM_WORKERS = os.cpu_count() // 4
UPDATE_LEARNING_RATE_PATIENCE = 10


for i in [1]:
    TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
    # ---------------------------
    # Load your data
    # ---------------------------
    df = pd.read_feather(f"data/sampled_dataset_{i}M.feather")
    # model_names = [
    #     "base",
    #     "denoising",
    #     "masked_denoising",
    #     "orthogonal",
    #     "weighted_base",
    #     "wassertein",
    # ]
    model_names = [
        "big_ae",
    ]
    print(f"Data Loaded - {i}M sampled data")

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

    # Convert to Torch tensor
    X_chunks = []

    for i in tqdm(range(0, len(df), CHUNK_SIZE)):
        chunk = df.iloc[i : i + CHUNK_SIZE][score_cols]
        X_chunk = chunk.apply(pd.to_numeric, errors="coerce").fillna(0).values
        X_chunks.append(X_chunk)

    X = np.vstack(X_chunks)
    X_tensor = torch.tensor(X, dtype=torch.float32)

    # Creating a training and validation dataset
    X_train, X_val = train_test_split(X_tensor, test_size=0.2, random_state=42)

    # Creating Dataset
    train_dataset = TensorDataset(X_train)
    val_dataset = TensorDataset(X_val)

    print("Tensor dataset created")

    # Creating the DataLoader objects
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        pin_memory=False,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        pin_memory=False,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        persistent_workers=True,
    )
    print("DataLoader completed")

    def train_and_evaluate_autoencoder(
        model_class,
        model_name,
        model_kwargs,
        train_loader,
        val_loader,
        X_tensor,
        df,
        score_cols,
        timestamp,
        early_stopping_patience,
        max_training_epochs,
        learning_rate,
        weight_decay,
        update_learning_rate_patience,
        corruption_level=None,
        corruption_value=None,
        requires_mask=False,
    ):
        """
        Unified function to train and evaluate any autoencoder model
        """
        print(f"\nTRAINING STARTING FOR {model_name.upper()} AUTOENCODER")

        # Create logger
        csv_logger = CSVLogger(save_dir="lightning_logs/", name=f"AE_TRAIN_{timestamp}")

        # Create early stopping callback
        early_stopping = EarlyStopping(
            monitor="val_loss",
            patience=early_stopping_patience,
            mode="min",
            verbose=True,
        )

        # Create model with appropriate parameters
        if corruption_level is not None and corruption_value is not None:
            model = model_class(
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                learning_patience=update_learning_rate_patience,
                corruption_level=corruption_level,
                corruption_value=corruption_value,
                **model_kwargs,
            )
        elif corruption_level is not None:
            model = model_class(
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                learning_patience=update_learning_rate_patience,
                corruption_level=corruption_level,
                **model_kwargs,
            )
        else:
            model = model_class(
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                learning_patience=update_learning_rate_patience,
                **model_kwargs,
            )

        # Configure model checkpointing
        checkpoint = ModelCheckpoint(
            mode="min",
            save_top_k=1,
            verbose=True,
            save_last=True,
            monitor="val_loss",
            filename="best_{epoch:02d}-{val_loss:.4f}",
        )

        # Create trainer
        trainer = pl.Trainer(
            logger=csv_logger,
            accelerator="auto",
            log_every_n_steps=1,
            max_epochs=max_training_epochs,
            callbacks=[early_stopping, checkpoint],
        )

        # Train the model
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

        # Load best model if available
        best_model_path = checkpoint.best_model_path
        if best_model_path:
            model = model_class.load_from_checkpoint(best_model_path)
            print(f"Loaded best model from: {best_model_path}")

        print("\nTraining complete!")

        # Create output directory
        output_dir = f"output/{timestamp}/{model_name}"
        os.makedirs(output_dir, exist_ok=True)

        # Post-training processing
        log_cleanup(
            os.path.join(csv_logger.log_dir, "metrics.csv"),
            f"{output_dir}/metrics.tsv",
        )

        # Handle model-specific input requirements for saving
        if requires_mask:
            zero_mask = torch.zeros_like(X_tensor)
            model_input = torch.cat([X_tensor, zero_mask], dim=1)
        else:
            model_input = X_tensor

        save_model(model, df, model_input, output_dir)

        # Calculate reconstruction metrics
        all_reconstructions = create_reconstructions(model, val_loader, requires_mask)

        # Calculate and save reconstruction metrics
        recon_metrics = calculate_reconstruction_metrics(
            X_val.numpy(),
            all_reconstructions.numpy(),
            f"{output_dir}/reconstruction_metrics.tsv",
            score_cols,
        )

        # Feature importance analysis
        importance_scores = feature_importance_analysis(
            model,
            val_loader,
            score_cols,
            f"{output_dir}/feature_importance.tsv",
            requires_mask,
        )

        # Create plots
        plot_training_metrics(
            len(df) / 1_000_000,
            f"{output_dir}/metrics.tsv",
            f"{output_dir}/training_metrics.png",
        )

        plot_correlation_comparison(
            len(df) / 1_000_000,
            score_cols,
            df,
            f"{output_dir}/composite_scores.feather",
            f"{output_dir}/score_correlation.png",
        )

        return model, recon_metrics, importance_scores

    # ---------------------------
    # Training Configuration
    # ---------------------------
    training_config = {
        "df": df,
        "X_tensor": X_tensor,
        "timestamp": TIMESTAMP,
        "score_cols": score_cols,
        "val_loader": val_loader,
        "weight_decay": WEIGHT_DECAY,
        "train_loader": train_loader,
        "learning_rate": LEARNING_RATE,
        "max_training_epochs": MAX_TRAINING_EPOCHS,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "update_learning_rate_patience": UPDATE_LEARNING_RATE_PATIENCE,
    }

    # ---------------------------
    # Train Base Autoencoder
    # ---------------------------
    # base_model, base_metrics, base_importance = train_and_evaluate_autoencoder(
    #     model_class=AutoEncoder, model_name="base", model_kwargs={}, **training_config
    # )

    # ---------------------------
    # Train Wassertein Autoencoder
    # ---------------------------
    base_model, base_metrics, base_importance = train_and_evaluate_autoencoder(
        model_class=BigAutoEncoder,
        model_name="big_ae",
        model_kwargs={},
        **training_config,
    )

    # ---------------------------
    # Train Weighted Base Autoencoder
    # ---------------------------
    # weight_base_model, weight_base_metrics, weight_base_importance = (
    #     train_and_evaluate_autoencoder(
    #         model_class=WeightedAutoEncoder,
    #         model_name="weighted_base",
    #         model_kwargs={
    #             "feature_weights": [
    #                 0.833764353,
    #                 1.590047852,
    #                 0.139906477,
    #                 0.07889904,
    #                 0.151527962,
    #                 0.839056569,
    #                 0.122767193,
    #                 0.819551701,
    #                 0.151148199,
    #                 0.305149593,
    #                 0.106692356,
    #                 8.281878519,
    #                 9.818368462,
    #                 9.351771899,
    #                 10.23376758,
    #                 12.47006061,
    #                 12.88421079,
    #                 0.223289636,
    #                 0.121357835,
    #                 0.121500688,
    #                 0.089116829,
    #                 0.108197796,
    #                 25.50816481,
    #                 0.269350337,
    #             ]
    #         },
    #         **training_config,
    #     )
    # )

    # ---------------------------
    # Train Denoising Autoencoder
    # ---------------------------
    # dae_model, dae_metrics, dae_importance = train_and_evaluate_autoencoder(
    #     model_kwargs={},
    #     model_name="denoising",
    #     model_class=DenoisingAutoEncoder,
    #     corruption_level=CORRUPTION_LEVEL,
    #     corruption_value=CORRUPTION_VALUE,
    #     **training_config,
    # )

    # ---------------------------
    # Train Masked Denoising Autoencoder
    # ---------------------------
    # masked_dae_model, masked_metrics, masked_importance = (
    #     train_and_evaluate_autoencoder(
    #         model_kwargs={},
    #         requires_mask=True,
    #         model_name="masked_denoising",
    #         corruption_level=CORRUPTION_LEVEL,
    #         model_class=MaskedDenoisingAutoEncoder,
    #         **training_config,
    #     )
    # )

    # ---------------------------
    # Train Orthogonal Autoencoder (if you have it)
    # ---------------------------
    # ortho_model, ortho_metrics, ortho_importance = train_and_evaluate_autoencoder(
    #     requires_mask=True,
    #     model_name="orthogonal",
    #     model_class=OrthogonalAutoEncoder,
    #     model_kwargs={"latent_dim": 3, "ortho_lambda": 1.0},
    #     **training_config,
    # )

    # 1. Comparative reconstruction metrics
    metrics_files = [
        f"output/{TIMESTAMP}/{name}/reconstruction_metrics.tsv" for name in model_names
    ]
    plot_reconstruction_metrics(
        metrics_files,
        model_names,
        f"output/{TIMESTAMP}/comparative_reconstruction_metrics.png",
    )

    # # 2. Feature importance comparison
    importance_files = [
        f"output/{TIMESTAMP}/{name}/feature_importance.tsv" for name in model_names
    ]
    plot_feature_importance(
        importance_files,
        model_names,
        score_cols,
        f"output/{TIMESTAMP}/comparative_feature_importance.png",
    )

    # # 3. Per-feature reconstruction quality (using R² as an example)
    per_feature_files = [
        f"output/{TIMESTAMP}/{name}/reconstruction_metrics_per_feature.tsv"
        for name in model_names
    ]

    # # You can create similar plots for other metrics like MSE, MAE, etc.
    for metric in ["MSE", "MAE", "Pearson_Correlation", "R2_Score"]:
        plot_per_feature_reconstruction(
            per_feature_files,
            model_names,
            score_cols,
            f"output/{TIMESTAMP}/comparative_per_feature_{metric.lower()}.png",
            metric_name=metric,
        )
