"""
=============================================================================
File 2 — Physics Loss Module
=============================================================================
Implements the Bergman Minimal Model ODE residuals as a differentiable
physics loss, shared across all three PINN architectures (RNN, LSTM,
Transformer).

Bergman Minimal Model:
  dG/dt = -(p1 + X)·G  +  p1·Gb  +  D(t)
  dX/dt = -p2·X  +  p3·I
  dI/dt = -n·I   +  γ·max(G - h, 0)  +  u(t)

Where:
  G    = blood glucose [mg/dL]
  X    = remote insulin effect (latent, never observed)
  I    = plasma insulin [U/min]
  D(t) = glucose appearance rate from meals [mg/dL/min]
  u(t) = exogenous insulin delivery [U/min]

Fixed literature parameters (Bergman 1981):
  p1    = 0.028    glucose effectiveness [1/min]
  p2    = 0.025    decay rate of remote insulin effect [1/min]
  p3    = 5e-5     insulin action on glucose uptake [1/(mU·min²)]
  n     = 0.09     insulin clearance rate [1/min]
  gamma = 0.006    pancreatic secretion gain [mU/(L·min²)]
  h     = 80.0     glucose secretion threshold [mg/dL]
  Gb    = 80.0     basal glucose [mg/dL]

These parameters are FIXED — they are never trained. Only the neural
network weights are updated. This is what separates PINN from the DP model.

Total loss:
  L_total = L_data + λ · L_physics

  L_data    = MSE(G_pred, G_true) + MSE(I_pred, I_true)
  L_physics = mean(r_G² + r_X² + r_I²)  at collocation points

λ is the physics loss coefficient — the hyperparameter we grid search over:
  λ ∈ [0.0, 0.001, 0.01, 0.1, 1.0]
  λ = 0.0 reduces to a plain data-driven model (no physics)

HOW THE PHYSICS LOSS WORKS FOR SEQUENCE MODELS:
  The RNN/LSTM/Transformer outputs a sequence of (G, X, I) predictions
  over the forecast horizon. The physics loss checks whether the temporal
  differences in those predictions are consistent with the Bergman ODEs.
  Specifically, finite differences approximate dG/dt, dX/dt, dI/dt from
  the predicted sequence, then the ODE residuals are computed at each step.
=============================================================================
"""

import torch
import torch.nn as nn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─────────────────────────────────────────────────────────────────────────────
# FIXED BERGMAN PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────

BERGMAN = {
    "p1":    0.028,    # glucose effectiveness [1/min]
    "p2":    0.025,    # decay rate of remote insulin effect X [1/min]
    "p3":    5.0e-5,   # insulin action on glucose uptake
    "n":     0.09,     # insulin clearance rate [1/min]
    "gamma": 0.006,    # pancreatic secretion gain
    "h":     80.0,     # glucose secretion threshold [mg/dL]
    "Gb":    80.0,     # basal glucose [mg/dL]
}

DT = 3.0   # timestep in minutes (confirmed from EDA: 3-min spacing)


# ─────────────────────────────────────────────────────────────────────────────
# ODE RESIDUALS — SEQUENCE VERSION
# ─────────────────────────────────────────────────────────────────────────────

def compute_physics_residuals(G_pred: torch.Tensor,
                               X_pred: torch.Tensor,
                               I_pred: torch.Tensor,
                               D:      torch.Tensor,
                               u:      torch.Tensor,
                               stats:  dict) -> torch.Tensor:
    """
    Compute Bergman ODE residuals over the predicted forecast horizon.

    All inputs are in NORMALISED space (z-scored). Predictions are
    inverse-transformed to physical units before applying the ODE,
    then residuals are computed in physical space.

    Args:
      G_pred : (B, H) normalised predicted glucose over horizon H
      X_pred : (B, H) predicted remote insulin effect (latent)
      I_pred : (B, H) normalised predicted insulin over horizon H
      D      : (B, H) meal disturbance [mg/dL/min] — physical units
      u      : (B, H) exogenous insulin [U/min] — physical units
      stats  : normalisation dict from data_ingestion.py

    Returns:
      residuals : (B, H-1, 3) — one residual per ODE per interior timestep
                  dim 2 = [r_G, r_X, r_I]
    """
    # ── Unpack normalisation constants ────────────────────────────────────────
    BG_mean  = stats["BG_mean"];   BG_std  = stats["BG_std"]
    I_mean   = stats["insulin_mean"]; I_std = stats["insulin_std"]

    # ── Unpack Bergman parameters ─────────────────────────────────────────────
    p1    = BERGMAN["p1"];   p2    = BERGMAN["p2"];  p3    = BERGMAN["p3"]
    n     = BERGMAN["n"];    gamma = BERGMAN["gamma"]
    h     = BERGMAN["h"];    Gb    = BERGMAN["Gb"]

    # ── Inverse-transform to physical units ───────────────────────────────────
    G_phys = G_pred * BG_std + BG_mean    # (B, H)  [mg/dL]
    I_phys = I_pred * I_std  + I_mean     # (B, H)  [U/min]
    # X is latent — no normalisation was applied, treat as physical directly
    X_phys = X_pred                        # (B, H)

    # ── Finite difference approximation of derivatives ────────────────────────
    # dG/dt ≈ (G[t+1] - G[t]) / dt   over the forecast horizon
    # Shape of each: (B, H-1)
    dGdt = (G_phys[:, 1:] - G_phys[:, :-1]) / DT
    dXdt = (X_phys[:, 1:] - X_phys[:, :-1]) / DT
    dIdt = (I_phys[:, 1:] - I_phys[:, :-1]) / DT

    # Use the left endpoint values for the ODE RHS (explicit Euler convention)
    G = G_phys[:, :-1]   # (B, H-1)
    X = X_phys[:, :-1]
    I = I_phys[:, :-1]
    D_t = D[:, :-1]      # meal disturbance at left endpoint
    u_t = u[:, :-1]      # insulin input at left endpoint

    # ── Bergman ODE right-hand sides ─────────────────────────────────────────
    # dG/dt = -(p1 + X)·G + p1·Gb + D(t)
    rhs_G = -(p1 + X) * G + p1 * Gb + D_t

    # dX/dt = -p2·X + p3·I
    rhs_X = -p2 * X + p3 * I

    # dI/dt = -n·I + γ·max(G - h, 0) + u(t)
    # torch.relu = max(·, 0) — avoids sympy/clamp conflict
    sec   = torch.relu(G - h)
    rhs_I = -n * I + gamma * sec + u_t

    # ── Residuals: (predicted derivative) - (ODE RHS) ─────────────────────────
    # If the model perfectly satisfies the physics, residuals = 0
    r_G = dGdt - rhs_G    # (B, H-1)
    r_X = dXdt - rhs_X    # (B, H-1)
    r_I = dIdt - rhs_I    # (B, H-1)

    # Stack to (B, H-1, 3)
    residuals = torch.stack([r_G, r_X, r_I], dim=-1)
    return residuals


# ─────────────────────────────────────────────────────────────────────────────
# COMBINED PINN LOSS
# ─────────────────────────────────────────────────────────────────────────────

def pinn_loss(G_pred:      torch.Tensor,
              X_pred:      torch.Tensor,
              I_pred:      torch.Tensor,
              G_true:      torch.Tensor,
              I_true:      torch.Tensor,
              D:           torch.Tensor,
              u:           torch.Tensor,
              stats:       dict,
              lambda_phys: float = 0.01) -> dict:
    """
    Compute total PINN loss for one batch.

    L_total = L_data + λ · L_physics

    L_data:
      MSE(G_pred, G_true) + MSE(I_pred, I_true)
      Both in normalised space. X is excluded — never observed,
      only constrained through the dX/dt physics residual.

    L_physics:
      Mean squared ODE residuals across all three equations and all
      interior timesteps of the forecast horizon.

    λ = lambda_phys:
      Hyperparameter controlling physics vs data trade-off.
      λ = 0.0  → pure data-driven (no physics penalty)
      λ = 1.0  → physics heavily enforced

    Args:
      G_pred  : (B, H) normalised predicted glucose
      X_pred  : (B, H) predicted latent insulin effect
      I_pred  : (B, H) normalised predicted insulin
      G_true  : (B, H) normalised true glucose
      I_true  : (B, H) normalised true insulin
      D       : (B, H) meal disturbance [mg/dL/min] physical units
      u       : (B, H) exogenous insulin [U/min] physical units
      stats   : normalisation dict
      lambda_phys : physics loss weight λ

    Returns:
      dict with keys: total, data, physics, G_mse, I_mse
    """
    mse = nn.MSELoss()

    # ── Data loss ─────────────────────────────────────────────────────────────
    loss_G    = mse(G_pred, G_true)
    loss_I    = mse(I_pred, I_true)
    loss_data = loss_G + loss_I

    # ── Physics loss ──────────────────────────────────────────────────────────
    if lambda_phys == 0.0:
        # Skip physics computation entirely when λ=0 — saves compute
        loss_phys = torch.tensor(0.0, device=G_pred.device)
    else:
        residuals = compute_physics_residuals(
            G_pred, X_pred, I_pred, D, u, stats)   # (B, H-1, 3)
        loss_phys = torch.mean(residuals ** 2)

        # Guard: if physics loss explodes early in training, skip this step
        if torch.isnan(loss_phys) or torch.isinf(loss_phys):
            loss_phys = torch.tensor(0.0, device=G_pred.device)

    # ── Total ─────────────────────────────────────────────────────────────────
    loss_total = loss_data + lambda_phys * loss_phys

    return {
        "total":   loss_total,
        "data":    loss_data.item(),
        "physics": loss_phys.item(),
        "G_mse":   loss_G.item(),
        "I_mse":   loss_I.item(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# STANDALONE DEMO
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Dummy batch to verify shapes and loss computation
    B, H = 32, 5   # batch size, horizon

    # Fake normalised predictions
    G_pred = torch.randn(B, H)
    X_pred = torch.randn(B, H) * 0.01
    I_pred = torch.randn(B, H)
    G_true = torch.randn(B, H)
    I_true = torch.randn(B, H)
    D      = torch.zeros(B, H)    # no meals in this dummy batch
    u      = torch.zeros(B, H)    # no exogenous insulin

    # Dummy stats matching expected format from data_ingestion.py
    stats = {
        "BG_mean": 126.97, "BG_std": 41.98,
        "insulin_mean": 0.0236, "insulin_std": 0.1771,
        "CHO_mean": 0.0, "CHO_std": 1.0,
    }

    for lam in [0.0, 0.001, 0.01, 0.1, 1.0]:
        losses = pinn_loss(G_pred, X_pred, I_pred, G_true, I_true,
                           D, u, stats, lambda_phys=lam)
        print(f"λ={lam:.3f}  |  total={losses['total']:.4f}  "
              f"data={losses['data']:.4f}  physics={losses['physics']:.4f}")
