"""
=============================================================================
File 4 — RNN-PINN
=============================================================================
Vanilla Recurrent Neural Network trained with the Bergman physics loss.

ARCHITECTURE:
  Input  : (B, 60, 3)  — 60 timesteps × [BG, insulin, CHO]
  RNN    : processes the sequence step-by-step with a single hidden state
  Output head : maps final hidden state → 3 × 5 values
              = (G, X, I) predictions for each of the 5 future timesteps

WHY RNN AS A SEPARATE MODEL FROM LSTM?
  The plain RNN is the simplest possible recurrent model. At each step:
    h_t = tanh(W_hh · h_{t-1} + W_xh · x_t + b)
  There are no gates — the hidden state is updated uniformly at every step.

  Compared to LSTM:
    - Fewer parameters (no input/forget/output gates, no cell state)
    - Faster to train
    - Known to struggle with long sequences due to vanishing gradients
    - For a 60-step window this is a real limitation — gradients from
      step 1 must propagate through 60 tanh operations, each of which
      compresses gradients, often to near zero by the time they reach
      early timesteps

  Including RNN lets us quantify: does the gating mechanism in LSTM
  actually help for this specific task and window length? If LSTM-PINN
  significantly outperforms RNN-PINN, the answer is yes.

ARCHITECTURE IS IDENTICAL TO LSTM-PINN EXCEPT:
  - nn.RNN instead of nn.LSTM
  - Single hidden state (no cell state c_n)
  - Same decoder head: hidden → (B, horizon * 3) → (B, horizon, 3)

PINN COMPONENT:
  Identical physics loss to LSTM-PINN — shared pinn_physics_loss.py module.
  λ is a hyperparameter grid searched identically.
=============================================================================
"""

import os
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
# MODEL
# ─────────────────────────────────────────────────────────────────────────────

class RNN_PINN(nn.Module):
    """
    Vanilla RNN encoder → linear decoder for 15-minute glucose forecasting.

    Args:
      input_size    : number of input features (3: BG, insulin, CHO)
      hidden_size   : RNN hidden state dimension
      num_layers    : number of stacked RNN layers
      output_horizon: number of future timesteps to predict (5 = 15 min)
      dropout       : dropout between RNN layers (only active if num_layers > 1)
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

        # Vanilla RNN — same interface as nn.LSTM but no cell state
        # nonlinearity='tanh': standard, smooth, differentiable activation
        self.rnn = nn.RNN(
            input_size  = input_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            batch_first = True,
            nonlinearity= "tanh",
            dropout     = dropout if num_layers > 1 else 0.0,
        )

        # Decoder: maps final hidden state → all future predictions
        # Identical structure to LSTM-PINN decoder for fair comparison
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
        # rnn_out : (B, 60, hidden_size) — all timestep hidden states
        # h_n     : (num_layers, B, hidden_size) — final hidden state
        rnn_out, h_n = self.rnn(x)

        # Take the last layer's final hidden state
        last_hidden = h_n[-1]   # (B, hidden_size)

        # Decode to (B, horizon, 3)
        out = self.decoder(last_hidden)
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
    Train the RNN-PINN. Identical training loop to LSTM-PINN for
    fair comparison — same optimizer, scheduler, gradient clipping,
    and physics loss formulation.
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
        path = os.path.join(save_dir, f"rnn_pinn_lambda{lambda_phys}.pt")
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
    fig.suptitle(f"RNN-PINN  λ={lambda_phys}", fontweight="bold")

    axes[0].semilogy(epochs, [h["total"]   for h in history], label="Total")
    axes[0].semilogy(epochs, [h["data"]    for h in history], label="Data")
    axes[0].semilogy(epochs, [h["physics"] for h in history], label="Physics")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss (log)")
    axes[0].set_title("Training Loss"); axes[0].legend()

    axes[1].plot(epochs, [h["val_rmse"] for h in history], color="darkorange")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("RMSE (mg/dL)")
    axes[1].set_title("Validation RMSE")

    plt.tight_layout()
    path = os.path.join(save_dir, f"rnn_pinn_lambda{lambda_phys}_history.png")
    plt.savefig(path, dpi=120); plt.show()
    print(f"Saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    SAVE_DIR    = "/content/drive/MyDrive/t1d_dataset/pinn_outputs/rnn"
    LAMBDA_PHYS = 0.01
    N_EPOCHS    = 50
    LR          = 1e-3
    HIDDEN_SIZE = 128
    NUM_LAYERS  = 2

    train_loader, val_loader, stats = load_pinn_data()

    model = RNN_PINN(
        input_size     = 3,
        hidden_size    = HIDDEN_SIZE,
        num_layers     = NUM_LAYERS,
        output_horizon = 5,
    ).to(DEVICE)

    print(f"RNN-PINN parameters: "
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
