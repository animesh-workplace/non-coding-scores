import os
import torch
import numpy as np
from tqdm import tqdm
import lightning as pl
from datetime import datetime
import fireducks.pandas as pd
from autoencoders.base_ae import AutoEncoder
from lightning.pytorch.loggers import CSVLogger
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from autoencoders.denoising_ae import DenoisingAutoEncoder
from autoencoders.orthogonal_ae import OrthogonalAutoEncoder
from autoencoders.binary_mask_dae import MaskedDenoisingAutoEncoder
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from utils import (
    log_cleanup,
    save_model,
    plot_training_metrics,
    plot_correlation_comparison,
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
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")


# ---------------------------
# Load your data
# ---------------------------
df = pd.read_feather("data/sampled_dataset_10M.feather")
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

# ---------------------------
# Training For BASE AUTOENCODER
# ---------------------------

print("\nTRAINING STARTING FOR BASE AUTOENCODER")
ae_csv_logger = CSVLogger(save_dir="lightning_logs/", name=f"AE_TRAIN_{TIMESTAMP}")
base_early_stopping = EarlyStopping(
    monitor="val_loss", patience=EARLY_STOPPING_PATIENCE, mode="min", verbose=True
)
ae_model = AutoEncoder(LEARNING_RATE, WEIGHT_DECAY, UPDATE_LEARNING_RATE_PATIENCE)
# Configure model checkpointing
ae_checkpoint = ModelCheckpoint(
    mode="min",
    save_top_k=1,
    verbose=True,
    save_last=True,
    monitor="val_loss",
    filename="best_{epoch:02d}-{val_loss:.4f}",
)
trainer = pl.Trainer(
    logger=ae_csv_logger,
    accelerator="auto",
    log_every_n_steps=1,
    max_epochs=MAX_TRAINING_EPOCHS,
    callbacks=[base_early_stopping, ae_checkpoint],
)
trainer.fit(ae_model, train_dataloaders=train_loader, val_dataloaders=val_loader)

print("\nTraining complete!")
log_cleanup(
    os.path.join(ae_csv_logger.log_dir, "metrics.csv"),
    f"output/{TIMESTAMP}/base/metrics.tsv",
)
save_model(ae_model, df, X_tensor, f"output/{TIMESTAMP}/base")
plot_training_metrics(
    len(df) / 1_000_000,
    f"output/{TIMESTAMP}/base/metrics.tsv",
    f"output/{TIMESTAMP}/base/training_metrics.png",
)
plot_correlation_comparison(
    len(df) / 1_000_000,
    score_cols,
    df,
    f"output/{TIMESTAMP}/base/composite_scores.feather",
    f"output/{TIMESTAMP}/base/score_correlation.png",
)

# ---------------------------
# Training For DENOISING AUTOENCODER
# ---------------------------

print("\nTRAINING STARTING FOR DESNOISING AUTOENCODER")
dae_csv_logger = CSVLogger(save_dir="lightning_logs/", name=f"AE_TRAIN_{TIMESTAMP}")
dae_early_stopping = EarlyStopping(
    monitor="val_loss", patience=EARLY_STOPPING_PATIENCE, mode="min", verbose=True
)
dae_model = DenoisingAutoEncoder(
    LEARNING_RATE,
    WEIGHT_DECAY,
    UPDATE_LEARNING_RATE_PATIENCE,
    CORRUPTION_LEVEL,
    CORRUPTION_VALUE,
)
# Configure model checkpointing
dae_checkpoint = ModelCheckpoint(
    mode="min",
    save_top_k=1,
    verbose=True,
    save_last=True,
    monitor="val_loss",
    filename="best_{epoch:02d}-{val_loss:.4f}",
)
trainer = pl.Trainer(
    logger=dae_csv_logger,
    accelerator="auto",
    log_every_n_steps=1,
    max_epochs=MAX_TRAINING_EPOCHS,
    callbacks=[dae_early_stopping, dae_checkpoint],
)
trainer.fit(dae_model, train_dataloaders=train_loader, val_dataloaders=val_loader)

best_model_path = dae_checkpoint.best_model_path
if best_model_path:
    dae_model = DenoisingAutoEncoder.load_from_checkpoint(best_model_path)
    print(f"Loaded best model from: {best_model_path}")

print("\nTraining complete!")
log_cleanup(
    os.path.join(dae_csv_logger.log_dir, "metrics.csv"),
    f"output/{TIMESTAMP}/denoising/metrics.tsv",
)
save_model(dae_model, df, X_tensor, f"output/{TIMESTAMP}/denoising")
plot_training_metrics(
    len(df) / 1_000_000,
    f"output/{TIMESTAMP}/denoising/metrics.tsv",
    f"output/{TIMESTAMP}/denoising/training_metrics.png",
)
plot_correlation_comparison(
    len(df) / 1_000_000,
    score_cols,
    df,
    f"output/{TIMESTAMP}/denoising/composite_scores.feather",
    f"output/{TIMESTAMP}/denoising/score_correlation.png",
)

# ---------------------------
# Training For DENOISING AUTOENCODER
# ---------------------------

print("\nTRAINING STARTING FOR MASKED DENOISING AUTOENCODER")
masked_dae_csv_logger = CSVLogger(
    save_dir="lightning_logs/", name=f"AE_TRAIN_{TIMESTAMP}"
)
masked_dae_early_stopping = EarlyStopping(
    monitor="val_loss", patience=EARLY_STOPPING_PATIENCE, mode="min", verbose=True
)
masked_dae_model = MaskedDenoisingAutoEncoder(
    LEARNING_RATE,
    WEIGHT_DECAY,
    UPDATE_LEARNING_RATE_PATIENCE,
    CORRUPTION_LEVEL,
)
# Configure model checkpointing
masked_dae_checkpoint = ModelCheckpoint(
    mode="min",
    save_top_k=1,
    verbose=True,
    save_last=True,
    monitor="val_loss",
    filename="best_{epoch:02d}-{val_loss:.4f}",
)
trainer = pl.Trainer(
    logger=masked_dae_csv_logger,
    accelerator="auto",
    log_every_n_steps=1,
    max_epochs=MAX_TRAINING_EPOCHS,
    callbacks=[masked_dae_early_stopping, masked_dae_checkpoint],
)
trainer.fit(
    masked_dae_model, train_dataloaders=train_loader, val_dataloaders=val_loader
)

best_model_path = masked_dae_checkpoint.best_model_path
if best_model_path:
    masked_dae_model = MaskedDenoisingAutoEncoder.load_from_checkpoint(best_model_path)
    print(f"Loaded best model from: {best_model_path}")

print("\nTraining complete!")
log_cleanup(
    os.path.join(masked_dae_csv_logger.log_dir, "metrics.csv"),
    f"output/{TIMESTAMP}/masked_denoising/metrics.tsv",
)
zero_mask = torch.zeros_like(X_tensor)
save_model(
    masked_dae_model,
    df,
    torch.cat([X_tensor, zero_mask], dim=1),
    f"output/{TIMESTAMP}/masked_denoising",
)
plot_training_metrics(
    len(df) / 1_000_000,
    f"output/{TIMESTAMP}/masked_denoising/metrics.tsv",
    f"output/{TIMESTAMP}/masked_denoising/training_metrics.png",
)
plot_correlation_comparison(
    len(df) / 1_000_000,
    score_cols,
    df,
    f"output/{TIMESTAMP}/masked_denoising/composite_scores.feather",
    f"output/{TIMESTAMP}/masked_denoising/score_correlation.png",
)

# ---------------------------
# Training For ORTHOGONAL AUTOENCODER
# ---------------------------

print("TRAINING STARTING FOR ORTHOGONAL AUTOENCODER")
ortho_ae_csv_logger = CSVLogger(
    save_dir="lightning_logs/", name=f"AE_TRAIN_{TIMESTAMP}"
)
ortho_ae_model = OrthogonalAutoEncoder(
    LEARNING_RATE,
    WEIGHT_DECAY,
    UPDATE_LEARNING_RATE_PATIENCE,
    0,
    latent_dim=3,  # You can adjust this
    ortho_lambda=1,  # You can adjust this
)
# Configure model checkpointing
ortho_ae_checkpoint = ModelCheckpoint(
    mode="min",
    save_top_k=1,
    verbose=True,
    save_last=True,
    monitor="val_loss",
    filename="best_{epoch:02d}-{val_loss:.4f}",
)
ortho_early_stopping = EarlyStopping(
    monitor="val_loss", patience=EARLY_STOPPING_PATIENCE, mode="min", verbose=True
)
trainer = pl.Trainer(
    logger=ortho_ae_csv_logger,
    accelerator="auto",
    log_every_n_steps=1,
    max_epochs=MAX_TRAINING_EPOCHS,
    callbacks=[ortho_early_stopping, ortho_ae_checkpoint],
)
trainer.fit(ortho_ae_model, train_dataloaders=train_loader, val_dataloaders=val_loader)

best_model_path = ortho_ae_checkpoint.best_model_path
if best_model_path:
    ortho_ae_model = OrthogonalAutoEncoder.load_from_checkpoint(best_model_path)
    print(f"Loaded best model from: {best_model_path}")

print("\nTraining complete!")
log_cleanup(
    os.path.join(ortho_ae_csv_logger.log_dir, "metrics.csv"),
    f"output/{TIMESTAMP}/orthogonal/metrics.tsv",
)
zero_mask = torch.zeros_like(X_tensor)
save_model(
    ortho_ae_model,
    df,
    torch.cat([X_tensor, zero_mask], dim=1),
    f"output/{TIMESTAMP}/orthogonal",
)
plot_training_metrics(
    len(df) / 1_000_000,
    f"output/{TIMESTAMP}/orthogonal/metrics.tsv",
    f"output/{TIMESTAMP}/orthogonal/training_metrics.png",
)
plot_correlation_comparison(
    len(df) / 1_000_000,
    score_cols,
    df,
    f"output/{TIMESTAMP}/orthogonal/composite_scores.feather",
    f"output/{TIMESTAMP}/orthogonal/score_correlation.png",
)

# Analyze latent dimensions
latent_analysis = ortho_ae_model.analyze_latent_dimensions(X_tensor, score_cols, df)
print("\n=== Latent Dimension Analysis ===")
for dim, results in latent_analysis.items():
    print(f"\n{dim.upper()}:")
    print("Top correlated features:")
    for feature, corr in results["top_features"]:
        print(f"  {feature}: {corr:.3f}")
