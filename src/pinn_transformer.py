"""
=============================================================================
File 5 — Transformer-PINN
=============================================================================
Transformer encoder trained with the Bergman physics loss.

ARCHITECTURE:
  Input  : (B, 60, 3)  — 60 timesteps × [BG, insulin, CHO]
  Positional Encoding → adds temporal ordering information
  Transformer Encoder → self-attention over the 60-step sequence
  Output head : maps CLS token (or mean-pooled) → 3 × 5 values
              = (G, X, I) predictions for each of the 5 future timesteps

WHY TRANSFORMER FOR GLUCOSE?
  Transformers process all 60 input timesteps simultaneously via
  self-attention, rather than sequentially like RNN/LSTM. This means:
    - No vanishing gradient problem — every timestep attends directly
      to every other timestep regardless of distance
    - The model can learn: "when I see a CHO spike at step t-30,
      BG typically rises at step t-20 and peaks at step t-10" by attending
      across arbitrary distances in the window
    - Parallelisable — all attention heads computed simultaneously on GPU

  For glucose specifically, attention can learn:
    - Pre-meal insulin bolus → glucose rise suppression (long-range)
    - CHO absorption curve shape (medium-range, 30-60 min patterns)
    - Basal insulin baseline (whole-window context)

TRANSFORMER vs LSTM for this task:
  - Transformer has no built-in sequential bias — temporal order is injected
    via positional encoding only. LSTM has sequential inductive bias built in.
  - For short horizons (15 min) LSTM's inductive bias may actually help.
  - Transformer typically needs more data to outperform LSTM — with 828k
    training windows this should not be a limiting factor.

POSITIONAL ENCODING:
  Standard sinusoidal encoding from "Attention Is All You Need" (Vaswani 2017).
  PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
  PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
  Added to the projected input embeddings before the attention layers.

PINN COMPONENT:
  Identical physics loss to RNN-PINN and LSTM-PINN — shared module.
=============================================================================
"""

import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from pinn_data_ingestion import load_pinn_data, DEVICE
from pinn_physics_loss   import pinn_loss

torch.manual_seed(42)
np.random.seed(42)


# ─────────────────────────────────────────────────────────────────────────────
# POSITIONAL ENCODING
# ─────────────────────────────────────────────────────────────────────────────

class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding — injects timestep ordering into the
    input embeddings since self-attention is permutation-invariant by default.

    For a sequence of length L with d_model dimensions:
      PE[:, 2i]   = sin(pos / 10000^(2i / d_model))
      PE[:, 2i+1] = cos(pos / 10000^(2i / d_model))

    The encoding is ADDED (not concatenated) to the input projection.
    """

    def __init__(self, d_model: int, max_len: int = 200, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Precompute the full positional encoding matrix
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float()
                        * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(pos * div)   # even indices → sin
        pe[:, 1::2] = torch.cos(pos * div)   # odd  indices → cos

        # Register as buffer — not a parameter, but moves to GPU with model
        pe = pe.unsqueeze(0)   # (1, max_len, d_model) for broadcasting
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# ─────────────────────────────────────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────────────────────────────────────

class Transformer_PINN(nn.Module):
    """
    Transformer encoder → linear decoder for 15-minute glucose forecasting.

    Args:
      input_size    : number of input features (3: BG, insulin, CHO)
      d_model       : transformer embedding dimension (must be divisible by nhead)
      nhead         : number of attention heads
      num_layers    : number of TransformerEncoderLayer stacks
      dim_feedforward: inner dimension of the FFN in each encoder layer
      output_horizon: number of future timesteps (5 = 15 min)
      dropout       : dropout rate throughout
    """

    def __init__(self,
                 input_size:      int = 3,
                 d_model:         int = 64,
                 nhead:           int = 4,
                 num_layers:      int = 2,
                 dim_feedforward: int = 256,
                 output_horizon:  int = 5,
                 dropout:         float = 0.1):
        super().__init__()
        self.output_horizon = output_horizon

        # Project raw 3-feature input to d_model dimensions
        # Required because Transformer attention operates in d_model space
        self.input_projection = nn.Linear(input_size, d_model)

        # Sinusoidal positional encoding
        self.pos_encoding = PositionalEncoding(
            d_model=d_model, max_len=200, dropout=dropout)

        # Transformer encoder stack
        encoder_layer = nn.TransformerEncoderLayer(
            d_model         = d_model,
            nhead           = nhead,
            dim_feedforward = dim_feedforward,
            dropout         = dropout,
            batch_first     = True,   # input shape: (B, seq, d_model)
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers)

        # Decoder: mean-pool over sequence → predict all future states
        # Mean pooling over the 60 timesteps gives a global sequence summary
        # that attends to the full context rather than just the last step
        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh(),
            nn.Linear(d_model, output_horizon * 3),
        )

    def forward(self, x: torch.Tensor) -> tuple:
        """
        Args:
          x : (B, 60, 3) — input window

        Returns:
          G_pred : (B, 5)  normalised glucose forecast
          X_pred : (B, 5)  latent insulin effect forecast
          I_pred : (B, 5)  normalised insulin forecast
        """
        # Project features to d_model: (B, 60, 3) → (B, 60, d_model)
        x = self.input_projection(x)

        # Add positional encoding
        x = self.pos_encoding(x)   # (B, 60, d_model)

        # Self-attention over all 60 timesteps simultaneously
        # Each position attends to all others — no sequential constraint
        enc = self.transformer_encoder(x)   # (B, 60, d_model)

        # Mean pool over the sequence dimension → global context vector
        pooled = enc.mean(dim=1)   # (B, d_model)

        # Decode to (B, horizon, 3)
        out = self.decoder(pooled)
        out = out.view(-1, self.output_horizon, 3)

        G_pred = out[:, :, 0]   # (B, 5)
        X_pred = out[:, :, 1]
        I_pred = out[:, :, 2]

        return G_pred, X_pred, I_pred


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────────────────────────────────────

def train(model:        nn.Module,
          train_loader,
          val_loader,
          stats:        dict,
          n_epochs:     int   = 50,
          lr:           float = 1e-3,
          lambda_phys:  float = 0.01,
          save_dir:     str   = None) -> list:
    """
    Train the Transformer-PINN. Identical training loop to RNN/LSTM-PINN
    for fair comparison.
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs, eta_min=1e-5)

    history = []

    for epoch in range(1, n_epochs + 1):
        model.train()
        epoch_losses = {"total": 0, "data": 0, "physics": 0,
                        "G_mse": 0, "I_mse": 0}
        n_batches = 0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            G_pred, X_pred, I_pred = model(X_batch)

            G_true = y_batch
            I_true = X_batch[:, -1, 1].unsqueeze(1).expand(-1, 5)

            B = X_batch.shape[0]
            D_phys = torch.zeros(B, 5, device=DEVICE)
            u_phys = torch.zeros(B, 5, device=DEVICE)

            losses = pinn_loss(
                G_pred, X_pred, I_pred,
                G_true, I_true,
                D_phys, u_phys,
                stats, lambda_phys=lambda_phys,
            )

            optimizer.zero_grad()
            losses["total"].backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            for k in epoch_losses:
                epoch_losses[k] += losses[k] if k == "total" \
                    else losses[k]
            n_batches += 1

        scheduler.step()

        for k in epoch_losses:
            epoch_losses[k] /= n_batches

        val_rmse = evaluate_rmse(model, val_loader, stats)
        epoch_losses["val_rmse"] = val_rmse
        history.append(epoch_losses)

        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d} | "
                  f"total={epoch_losses['total']:.4e} | "
                  f"data={epoch_losses['data']:.4e} | "
                  f"physics={epoch_losses['physics']:.4e} | "
                  f"val_RMSE={val_rmse:.2f} mg/dL | "
                  f"λ={lambda_phys}")

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir,
                            f"transformer_pinn_lambda{lambda_phys}.pt")
        torch.save(model.state_dict(), path)
        print(f"Saved → {path}")

    return history


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_rmse(model, val_loader, stats: dict) -> float:
    model.eval()
    sq_errors, n = 0.0, 0

    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            G_pred, _, _ = model(X_batch)

            G_pred_phys = G_pred * stats["BG_std"] + stats["BG_mean"]
            G_true_phys = y_batch * stats["BG_std"] + stats["BG_mean"]

            sq_errors += ((G_pred_phys - G_true_phys) ** 2).sum().item()
            n         += G_pred_phys.numel()

    model.train()
    return float(np.sqrt(sq_errors / n))


def plot_history(history: list, lambda_phys: float, save_dir: str):
    epochs = range(1, len(history) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    fig.suptitle(f"Transformer-PINN  λ={lambda_phys}", fontweight="bold")

    axes[0].semilogy(epochs, [h["total"]   for h in history], label="Total")
    axes[0].semilogy(epochs, [h["data"]    for h in history], label="Data")
    axes[0].semilogy(epochs, [h["physics"] for h in history], label="Physics")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss (log)")
    axes[0].set_title("Training Loss"); axes[0].legend()

    axes[1].plot(epochs, [h["val_rmse"] for h in history], color="green")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("RMSE (mg/dL)")
    axes[1].set_title("Validation RMSE")

    plt.tight_layout()
    path = os.path.join(save_dir,
                        f"transformer_pinn_lambda{lambda_phys}_history.png")
    plt.savefig(path, dpi=120); plt.show()
    print(f"Saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    SAVE_DIR        = "/content/drive/MyDrive/t1d_dataset/pinn_outputs/transformer"
    LAMBDA_PHYS     = 0.01
    N_EPOCHS        = 50
    LR              = 1e-3
    D_MODEL         = 64
    NHEAD           = 4
    NUM_LAYERS      = 2
    DIM_FEEDFORWARD = 256

    train_loader, val_loader, stats = load_pinn_data()

    model = Transformer_PINN(
        input_size      = 3,
        d_model         = D_MODEL,
        nhead           = NHEAD,
        num_layers      = NUM_LAYERS,
        dim_feedforward = DIM_FEEDFORWARD,
        output_horizon  = 5,
    ).to(DEVICE)

    print(f"Transformer-PINN parameters: "
          f"{sum(p.numel() for p in model.parameters()):,}")

    history = train(
        model, train_loader, val_loader, stats,
        n_epochs    = N_EPOCHS,
        lr          = LR,
        lambda_phys = LAMBDA_PHYS,
        save_dir    = SAVE_DIR,
    )

    final_rmse = evaluate_rmse(model, val_loader, stats)
    print(f"\nFinal Val RMSE: {final_rmse:.3f} mg/dL")
    plot_history(history, LAMBDA_PHYS, SAVE_DIR)
