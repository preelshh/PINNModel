"""
=============================================================================
File 3 — LSTM-PINN
=============================================================================
Long Short-Term Memory network trained with the Bergman physics loss.

ARCHITECTURE:
  Input  : (B, 60, 3)  — 60 timesteps × [BG, insulin, CHO]
  LSTM   : processes the 60-step sequence, maintaining hidden state
           that captures long-range glucose dynamics across the window
  Output head : maps final hidden state → 3 × 5 values
              = (G, X, I) predictions for each of the 5 future timesteps

WHY LSTM FOR GLUCOSE?
  LSTMs have gating mechanisms (input gate, forget gate, output gate) that
  let the model selectively remember or forget past information. For glucose
  dynamics this is important because:
    - Basal insulin delivery is slow and persistent → needs long memory
    - Meal spikes are sharp and transient → need selective forgetting after
    - The hidden state acts as a learned approximation of (G, X, I) state

LSTM vs RNN:
  Plain RNNs suffer from vanishing gradients over long sequences — after
  60 timesteps of backprop, gradients from early timesteps shrink to zero.
  LSTMs solve this with the cell state (a "conveyor belt" of memory) that
  gradients can flow through without vanishing. For a 3-hour window this
  matters substantially.

PINN COMPONENT:
  The output head predicts [G, X, I] for all 5 future steps, not just G.
  This is required by the physics loss — all three Bergman state variables
  must be predicted to compute ODE residuals. Only G and I are supervised
  by data loss. X is only constrained by the dX/dt physics residual.

OUTPUT STRUCTURE:
  The decoder maps LSTM hidden state → (B, 5, 3):
    dim 0 = batch
    dim 1 = forecast timestep (0..4 = t+3min .. t+15min)
    dim 2 = state variable (0=G, 1=X, 2=I)
=============================================================================
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from pinn_data_ingestion import load_pinn_data, inverse_norm_BG, DEVICE
from pinn_physics_loss   import pinn_loss

torch.manual_seed(42)
np.random.seed(42)


# ─────────────────────────────────────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────────────────────────────────────

class LSTM_PINN(nn.Module):
    """
    LSTM encoder → linear decoder for 15-minute glucose forecasting.

    Args:
      input_size    : number of input features (3: BG, insulin, CHO)
      hidden_size   : LSTM hidden state dimension
      num_layers    : number of stacked LSTM layers
      output_horizon: number of future timesteps to predict (5 = 15 min)
      dropout       : dropout between LSTM layers (only active if num_layers > 1)
    """

    def __init__(self,
                 input_size:     int = 3,
                 hidden_size:    int = 128,
                 num_layers:     int = 2,
                 output_horizon: int = 5,
                 dropout:        float = 0.1):
        super().__init__()
        self.hidden_size    = hidden_size
        self.num_layers     = num_layers
        self.output_horizon = output_horizon

        # LSTM encoder — processes the full 60-step input window
        # batch_first=True: input shape is (B, seq_len, features)
        self.lstm = nn.LSTM(
            input_size  = input_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = dropout if num_layers > 1 else 0.0,
        )

        # Decoder: maps the final hidden state → all future predictions
        # Output: (B, output_horizon * 3) — G, X, I for each future step
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, output_horizon * 3),
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
        # Run LSTM over the 60-step window
        # lstm_out: (B, 60, hidden_size) — all hidden states
        # h_n     : (num_layers, B, hidden_size) — final hidden state
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Take the last layer's final hidden state as the sequence summary
        last_hidden = h_n[-1]   # (B, hidden_size)

        # Decode to (B, horizon * 3) then reshape to (B, horizon, 3)
        out = self.decoder(last_hidden)
        out = out.view(-1, self.output_horizon, 3)   # (B, 5, 3)

        G_pred = out[:, :, 0]   # (B, 5) — glucose
        X_pred = out[:, :, 1]   # (B, 5) — latent X
        I_pred = out[:, :, 2]   # (B, 5) — insulin

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
    Train the LSTM-PINN for n_epochs.

    D(t) and u(t) for the physics loss are extracted from the input window's
    CHO and insulin columns (indices 2 and 1 in the feature list).
    The forecast horizon's D and u are approximated as zero — standard
    assumption since future meal/insulin delivery is unknown at inference time.

    Returns history: list of per-epoch dicts with loss components.
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
            X_batch = X_batch.to(DEVICE)   # (B, 60, 3)
            y_batch = y_batch.to(DEVICE)   # (B, 5)  — normalised BG only

            # Forward pass
            G_pred, X_pred, I_pred = model(X_batch)   # each (B, 5)

            # Ground truth: y_batch is BG only
            G_true = y_batch   # (B, 5)

            # For I_true: use last insulin value of input window repeated
            # across horizon — best available approximation
            I_true = X_batch[:, -1, 1].unsqueeze(1).expand(-1, 5)  # (B, 5)

            # D and u for physics loss: zeros over forecast horizon
            # (future meals/insulin are unknown at inference time)
            B = X_batch.shape[0]
            D_phys = torch.zeros(B, 5, device=DEVICE)
            u_phys = torch.zeros(B, 5, device=DEVICE)

            # Compute PINN loss
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

        # Average losses over batches
        for k in epoch_losses:
            epoch_losses[k] /= n_batches

        # Validation RMSE
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
        path = os.path.join(save_dir, f"lstm_pinn_lambda{lambda_phys}.pt")
        torch.save(model.state_dict(), path)
        print(f"Saved → {path}")

    return history


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_rmse(model, val_loader, stats: dict) -> float:
    """Compute validation RMSE in mg/dL (physical units)."""
    model.eval()
    sq_errors, n = 0.0, 0

    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            G_pred, _, _ = model(X_batch)   # (B, 5) normalised

            # Inverse transform both to mg/dL
            G_pred_phys = G_pred * stats["BG_std"] + stats["BG_mean"]
            G_true_phys = y_batch * stats["BG_std"] + stats["BG_mean"]

            sq_errors += ((G_pred_phys - G_true_phys) ** 2).sum().item()
            n         += G_pred_phys.numel()

    model.train()
    return float(np.sqrt(sq_errors / n))


def plot_history(history: list, lambda_phys: float, save_dir: str):
    epochs = range(1, len(history) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    fig.suptitle(f"LSTM-PINN  λ={lambda_phys}", fontweight="bold")

    axes[0].semilogy(epochs, [h["total"]   for h in history], label="Total")
    axes[0].semilogy(epochs, [h["data"]    for h in history], label="Data")
    axes[0].semilogy(epochs, [h["physics"] for h in history], label="Physics")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss (log)")
    axes[0].set_title("Training Loss"); axes[0].legend()

    axes[1].plot(epochs, [h["val_rmse"] for h in history], color="steelblue")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("RMSE (mg/dL)")
    axes[1].set_title("Validation RMSE")

    plt.tight_layout()
    path = os.path.join(save_dir, f"lstm_pinn_lambda{lambda_phys}_history.png")
    plt.savefig(path, dpi=120); plt.show()
    print(f"Saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    SAVE_DIR     = "/content/drive/MyDrive/t1d_dataset/pinn_outputs/lstm"
    LAMBDA_PHYS  = 0.01     # single run — use hparam_search.py for full grid
    N_EPOCHS     = 50
    LR           = 1e-3
    HIDDEN_SIZE  = 128
    NUM_LAYERS   = 2

    train_loader, val_loader, stats = load_pinn_data()

    model = LSTM_PINN(
        input_size     = 3,
        hidden_size    = HIDDEN_SIZE,
        num_layers     = NUM_LAYERS,
        output_horizon = 5,
    ).to(DEVICE)

    print(f"LSTM-PINN parameters: "
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
