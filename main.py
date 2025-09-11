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
from lightning.pytorch.callbacks import EarlyStopping
from torch.utils.data import DataLoader, TensorDataset
from lightning.pytorch.callbacks import LearningRateMonitor

# --- Configuration ---
BATCH_SIZE = 16384
CHUNK_SIZE = 100000
WEIGHT_DECAY = 1e-5
LEARNING_RATE = 1e-3
MAX_TRAINING_EPOCHS = 500
EARLY_STOPPING_PATIENCE = 20
NUM_WORKERS = os.cpu_count() // 4
UPDATE_LEARNING_RATE_PATIENCE = 10
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")


# ---------------------------
# Load your data
# ---------------------------
df = pd.read_feather("data/sample_dataset_1M.feather")
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

print("TRAINING STARTING FOR BASE AUTOENCODER")
# The CSVLogger will save all metrics logged with self.log() to a CSV file.
csv_logger = CSVLogger(save_dir="lightning_logs/", name=f"AE_TRAIN_{TIMESTAMP}")
lr_monitor = LearningRateMonitor(logging_interval="epoch")

model = AutoEncoder()
early_stopping = EarlyStopping(
    monitor="val_loss", patience=EARLY_STOPPING_PATIENCE, mode="min", verbose=True
)
trainer = pl.Trainer(
    logger=csv_logger,
    accelerator="auto",
    log_every_n_steps=1,
    callbacks=[early_stopping],
    max_epochs=MAX_TRAINING_EPOCHS,
)
trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

print("\nTraining complete!")
print(f"Metrics have been saved to: {os.path.join(csv_logger.log_dir, 'metrics.csv')}")
