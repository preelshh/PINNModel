"""
=============================================================================
File 1 — Data Ingestion
=============================================================================
Loads train_dataset.csv and val_dataset.csv and returns sliding-window
tensors ready for RNN / LSTM / Transformer PINN training.

Confirmed data facts (from EDA):
  - 864,600 train rows / 216,150 val rows
  - 600 train trajectories / 150 val trajectories
  - Every trajectory is exactly 1441 timesteps, 3-minute spacing
  - Trajectories are contiguous blocks in the CSV (confirmed)
  - Columns used: BG, insulin, CHO
  - BG outliers: min=-0.00, max=787.68  → clip to [30, 400]

Windowing:
  - Input  : 60 timesteps = 3 hours of [BG, insulin, CHO]
  - Output : 5  timesteps = 15 minutes of BG only

Output shapes:
  X : (N_windows, 60, 3)   float32
  y : (N_windows, 5)       float32  — BG only, in normalised units
=============================================================================
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE      = "/content/drive/MyDrive/t1d_dataset"
TRAIN_CSV = f"{BASE}/train_dataset.csv"
VAL_CSV   = f"{BASE}/val_dataset.csv"

# ── Window configuration ──────────────────────────────────────────────────────
INPUT_WINDOW  = 60   # timesteps of history fed to the model (3 hours)
OUTPUT_HORIZON = 5   # timesteps to predict ahead (15 minutes)
FEATURES      = ["BG", "insulin", "CHO"]   # input feature columns
N_FEATURES    = len(FEATURES)              # 3

# ── Data quality ──────────────────────────────────────────────────────────────
BG_CLIP_MIN = 30.0
BG_CLIP_MAX = 400.0


# ─────────────────────────────────────────────────────────────────────────────
# CLEANING
# ─────────────────────────────────────────────────────────────────────────────

def _clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["BG"]      = df["BG"].clip(lower=BG_CLIP_MIN, upper=BG_CLIP_MAX)
    df["insulin"] = df["insulin"].fillna(0.0)
    df["CHO"]     = df["CHO"].fillna(0.0)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# NORMALISATION
# ─────────────────────────────────────────────────────────────────────────────

def compute_norm_stats(df: pd.DataFrame) -> dict:
    """
    Compute normalisation statistics from training data only.
    All three features are z-scored independently.
    Stats are computed on TRAINING SET only — applied to val without refitting.
    """
    stats = {}
    for col in FEATURES:
        stats[f"{col}_mean"] = float(df[col].mean())
        stats[f"{col}_std"]  = float(df[col].std()) + 1e-8
    return stats


def apply_norm(df: pd.DataFrame, stats: dict) -> pd.DataFrame:
    df = df.copy()
    for col in FEATURES:
        df[col] = (df[col] - stats[f"{col}_mean"]) / stats[f"{col}_std"]
    return df


def inverse_norm_BG(arr: np.ndarray, stats: dict) -> np.ndarray:
    """Convert normalised BG predictions back to mg/dL."""
    return arr * stats["BG_std"] + stats["BG_mean"]


# ─────────────────────────────────────────────────────────────────────────────
# SLIDING WINDOW EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def extract_windows(df: pd.DataFrame,
                    input_window:   int = INPUT_WINDOW,
                    output_horizon: int = OUTPUT_HORIZON) -> tuple:
    """
    Slide a window across each trajectory to produce (X, y) pairs.

    Each trajectory is 1441 steps. For one trajectory:
      - Valid window start positions: 0 to (1441 - input_window - output_horizon)
      - = 0 to 1376 → 1376 windows per trajectory

    Windows never cross trajectory boundaries — each group is processed
    independently, so the model never sees across patient-scenario boundaries.

    Returns:
      X : (N_total_windows, input_window, n_features)   float32
      y : (N_total_windows, output_horizon)              float32  — normalised BG
    """
    X_list, y_list = [], []

    # Data is confirmed contiguous — group directly without sorting
    grouped = df.groupby(["patient_id", "scenario"], sort=False)

    for _, grp in grouped:
        # Extract feature matrix for this trajectory — shape (1441, 3)
        vals = grp[FEATURES].values.astype(np.float32)
        # BG column index is 0 in FEATURES list
        bg_idx = FEATURES.index("BG")

        n_steps = len(vals)
        max_start = n_steps - input_window - output_horizon

        for start in range(max_start + 1):
            end     = start + input_window
            fut_end = end   + output_horizon

            X_list.append(vals[start:end])                    # (60, 3)
            y_list.append(vals[end:fut_end, bg_idx])          # (5,)  BG only

    X = np.stack(X_list, axis=0)   # (N, 60, 3)
    y = np.stack(y_list, axis=0)   # (N, 5)

    return X, y


# ─────────────────────────────────────────────────────────────────────────────
# PYTORCH DATASET
# ─────────────────────────────────────────────────────────────────────────────

class GlucoseWindowDataset(Dataset):
    """
    PyTorch Dataset wrapping (X, y) window arrays.
    X shape : (N, 60, 3)
    y shape : (N, 5)
    """
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def load_pinn_data(train_csv:   str = TRAIN_CSV,
                   val_csv:     str = VAL_CSV,
                   batch_size:  int = 512,
                   num_workers: int = 2) -> tuple:
    """
    Full pipeline: load → clean → normalise → window → DataLoader.

    Returns:
      train_loader : DataLoader  (X: (B,60,3), y: (B,5))
      val_loader   : DataLoader
      stats        : dict of normalisation constants (for inverse transform)
    """
    print("Loading CSVs …")
    df_tr = pd.read_csv(train_csv)
    df_va = pd.read_csv(val_csv)
    print(f"  Train raw: {df_tr.shape}  |  Val raw: {df_va.shape}")

    # Clean
    df_tr = _clean(df_tr)
    df_va = _clean(df_va)

    # Fit normalisation stats on train only
    stats = compute_norm_stats(df_tr)

    # Normalise both splits using train stats
    df_tr_n = apply_norm(df_tr, stats)
    df_va_n = apply_norm(df_va, stats)

    # Extract sliding windows
    print("Extracting windows …")
    X_tr, y_tr = extract_windows(df_tr_n)
    X_va, y_va = extract_windows(df_va_n)
    print(f"  Train windows: {X_tr.shape}  |  Val windows: {X_va.shape}")

    # Build DataLoaders
    train_ds     = GlucoseWindowDataset(X_tr, y_tr)
    val_ds       = GlucoseWindowDataset(X_va, y_va)
    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  num_workers=num_workers,
                              pin_memory=(DEVICE.type == "cuda"))
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, num_workers=num_workers,
                              pin_memory=(DEVICE.type == "cuda"))

    # Summary
    tir_raw = ((df_tr["BG"] >= 70) & (df_tr["BG"] <= 180)).mean() * 100
    print(f"\n── Ingestion summary ─────────────────────────────")
    print(f"  Train windows    : {len(train_ds):,}")
    print(f"  Val   windows    : {len(val_ds):,}")
    print(f"  Input shape      : (batch, {INPUT_WINDOW}, {N_FEATURES})")
    print(f"  Output shape     : (batch, {OUTPUT_HORIZON})")
    print(f"  BG  mean/std     : {stats['BG_mean']:.1f} / {stats['BG_std']:.1f} mg/dL")
    print(f"  Ins mean/std     : {stats['insulin_mean']:.4f} / {stats['insulin_std']:.4f}")
    print(f"  TIR (train raw)  : {tir_raw:.1f}%")
    print(f"──────────────────────────────────────────────────")

    return train_loader, val_loader, stats


# ── Sanity check ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    train_loader, val_loader, stats = load_pinn_data()
    X_batch, y_batch = next(iter(train_loader))
    print(f"\nSample batch:")
    print(f"  X : {X_batch.shape}   dtype={X_batch.dtype}")
    print(f"  y : {y_batch.shape}   dtype={y_batch.dtype}")
    print(f"  Normalisation stats: {stats}")
