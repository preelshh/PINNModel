"""
=============================================================================
File 6 — Hyperparameter Grid Search
=============================================================================
Runs all three PINN models (RNN, LSTM, Transformer) across the full
lambda grid and produces a unified results table and comparison plots.

Lambda grid: [0.0, 0.001, 0.01, 0.1, 1.0]
  λ = 0.0  → pure data-driven (physics loss disabled)
  λ = 0.001 → very light physics regularisation
  λ = 0.01  → moderate physics (recommended starting point)
  λ = 0.1   → strong physics
  λ = 1.0   → physics dominates — may overfit to fixed Bergman parameters

Total runs: 3 models × 5 lambda values = 15 training runs

Results saved to:
  pinn_outputs/hparam_search/
    results_table.csv          — RMSE per model per lambda
    hparam_comparison.png      — bar chart: model × lambda
    best_model_summary.txt     — best config per model + overall winner
=============================================================================
"""

import os
import json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from pinn_data_ingestion import load_pinn_data, DEVICE
from pinn_lstm        import LSTM_PINN,        train as train_lstm
from pinn_rnn         import RNN_PINN,         train as train_rnn
from pinn_transformer import Transformer_PINN, train as train_transformer

torch.manual_seed(42)
np.random.seed(42)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

SAVE_DIR     = "/content/drive/MyDrive/t1d_dataset/pinn_outputs/hparam_search"
LAMBDA_GRID  = [0.0, 0.001, 0.01, 0.1, 1.0]
N_EPOCHS     = 50      # increase to 100+ for final submission
LR           = 1e-3
BATCH_SIZE   = 512

# Architecture hyperparameters — kept fixed across the λ grid
# so that only λ varies between runs
MODEL_CONFIGS = {
    "RNN": {
        "hidden_size": 128,
        "num_layers":  2,
    },
    "LSTM": {
        "hidden_size": 128,
        "num_layers":  2,
    },
    "Transformer": {
        "d_model":         64,
        "nhead":           4,
        "num_layers":      2,
        "dim_feedforward": 256,
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# MODEL FACTORY
# ─────────────────────────────────────────────────────────────────────────────

def build_model(model_name: str) -> torch.nn.Module:
    cfg = MODEL_CONFIGS[model_name]
    if model_name == "RNN":
        return RNN_PINN(
            input_size     = 3,
            hidden_size    = cfg["hidden_size"],
            num_layers     = cfg["num_layers"],
            output_horizon = 5,
        ).to(DEVICE)
    elif model_name == "LSTM":
        return LSTM_PINN(
            input_size     = 3,
            hidden_size    = cfg["hidden_size"],
            num_layers     = cfg["num_layers"],
            output_horizon = 5,
        ).to(DEVICE)
    elif model_name == "Transformer":
        return Transformer_PINN(
            input_size      = 3,
            d_model         = cfg["d_model"],
            nhead           = cfg["nhead"],
            num_layers      = cfg["num_layers"],
            dim_feedforward = cfg["dim_feedforward"],
            output_horizon  = 5,
        ).to(DEVICE)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def get_train_fn(model_name: str):
    return {"RNN": train_rnn, "LSTM": train_lstm,
            "Transformer": train_transformer}[model_name]


def get_eval_fn(model_name: str):
    from pinn_lstm        import evaluate_rmse as eval_lstm
    from pinn_rnn         import evaluate_rmse as eval_rnn
    from pinn_transformer import evaluate_rmse as eval_transformer
    return {"RNN": eval_rnn, "LSTM": eval_lstm,
            "Transformer": eval_transformer}[model_name]


# ─────────────────────────────────────────────────────────────────────────────
# GRID SEARCH
# ─────────────────────────────────────────────────────────────────────────────

def run_grid_search():
    os.makedirs(SAVE_DIR, exist_ok=True)

    print("Loading data …")
    train_loader, val_loader, stats = load_pinn_data(batch_size=BATCH_SIZE)

    results = []   # list of dicts: {model, lambda, val_rmse, history}
    model_names = list(MODEL_CONFIGS.keys())

    total_runs = len(model_names) * len(LAMBDA_GRID)
    run = 0

    for model_name in model_names:
        for lam in LAMBDA_GRID:
            run += 1
            print(f"\n{'='*60}")
            print(f"  Run {run}/{total_runs}: {model_name}  λ={lam}")
            print(f"{'='*60}")

            model    = build_model(model_name)
            train_fn = get_train_fn(model_name)
            eval_fn  = get_eval_fn(model_name)

            n_params = sum(p.numel() for p in model.parameters())
            print(f"  Parameters: {n_params:,}")

            # Train
            history = train_fn(
                model, train_loader, val_loader, stats,
                n_epochs    = N_EPOCHS,
                lr          = LR,
                lambda_phys = lam,
                save_dir    = os.path.join(SAVE_DIR, model_name.lower()),
            )

            # Final validation RMSE in mg/dL
            final_rmse = eval_fn(model, val_loader, stats)

            print(f"\n  → {model_name}  λ={lam}  Final Val RMSE: "
                  f"{final_rmse:.3f} mg/dL")

            results.append({
                "model":       model_name,
                "lambda":      lam,
                "val_rmse":    final_rmse,
                "best_epoch":  int(np.argmin([h["val_rmse"] for h in history])) + 1,
                "best_rmse":   float(np.min([h["val_rmse"] for h in history])),
                "n_params":    n_params,
            })

            # Save per-run history
            hist_path = os.path.join(
                SAVE_DIR, f"{model_name.lower()}_lambda{lam}_history.json")
            with open(hist_path, "w") as f:
                json.dump(history, f)

            # Clear GPU memory between runs
            del model
            torch.cuda.empty_cache() if DEVICE.type == "cuda" else None

    return results, stats


# ─────────────────────────────────────────────────────────────────────────────
# RESULTS TABLE & PLOTS
# ─────────────────────────────────────────────────────────────────────────────

def save_results(results: list):
    df = pd.DataFrame(results)

    # ── Results table ──────────────────────────────────────────────────────
    table_path = os.path.join(SAVE_DIR, "results_table.csv")
    df.to_csv(table_path, index=False)
    print(f"\nResults table saved → {table_path}")

    # Pretty-print pivot table
    pivot = df.pivot_table(
        index="model", columns="lambda", values="val_rmse").round(3)
    print("\n── Val RMSE (mg/dL) by model × lambda ──")
    print(pivot.to_string())

    # ── Bar chart: model × lambda ─────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 5))
    model_names = df["model"].unique()
    x = np.arange(len(LAMBDA_GRID))
    w = 0.25
    colors = {"RNN": "#4C72B0", "LSTM": "#DD8452", "Transformer": "#55A868"}

    for i, name in enumerate(model_names):
        sub   = df[df["model"] == name].sort_values("lambda")
        rmses = sub["val_rmse"].values
        ax.bar(x + i * w, rmses, w,
               label=name, color=colors.get(name, "gray"), alpha=0.85)

    ax.set_xticks(x + w)
    ax.set_xticklabels([str(l) for l in LAMBDA_GRID])
    ax.set_xlabel("Physics Loss Coefficient λ")
    ax.set_ylabel("Validation RMSE (mg/dL)")
    ax.set_title("PINN Model Comparison: RMSE vs λ", fontweight="bold")
    ax.legend()
    plt.tight_layout()
    plot_path = os.path.join(SAVE_DIR, "hparam_comparison.png")
    plt.savefig(plot_path, dpi=150)
    plt.show()
    print(f"Plot saved → {plot_path}")

    # ── RMSE vs lambda line plot per model ────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    for name in model_names:
        sub = df[df["model"] == name].sort_values("lambda")
        ax.plot(sub["lambda"], sub["val_rmse"],
                marker="o", label=name, color=colors.get(name, "gray"))
    ax.set_xscale("symlog", linthresh=1e-3)
    ax.set_xlabel("λ (log scale)")
    ax.set_ylabel("Validation RMSE (mg/dL)")
    ax.set_title("RMSE vs Physics Loss Weight", fontweight="bold")
    ax.legend()
    plt.tight_layout()
    line_path = os.path.join(SAVE_DIR, "rmse_vs_lambda.png")
    plt.savefig(line_path, dpi=150)
    plt.show()
    print(f"Plot saved → {line_path}")

    # ── Best model summary ─────────────────────────────────────────────────
    best_per_model = df.loc[df.groupby("model")["val_rmse"].idxmin()]
    overall_best   = df.loc[df["val_rmse"].idxmin()]

    summary_path = os.path.join(SAVE_DIR, "best_model_summary.txt")
    with open(summary_path, "w") as f:
        f.write("PINN Hyperparameter Search — Best Configurations\n")
        f.write("=" * 50 + "\n\n")
        for _, row in best_per_model.iterrows():
            f.write(f"{row['model']:15s}  best λ={row['lambda']}  "
                    f"RMSE={row['val_rmse']:.3f} mg/dL  "
                    f"(best epoch {row['best_epoch']})\n")
        f.write(f"\nOverall best: {overall_best['model']}  "
                f"λ={overall_best['lambda']}  "
                f"RMSE={overall_best['val_rmse']:.3f} mg/dL\n")

    print(f"\nBest model summary saved → {summary_path}")
    print(f"\n── Best per model ──")
    print(best_per_model[["model", "lambda", "val_rmse",
                           "best_epoch"]].to_string(index=False))
    print(f"\n── Overall winner ──")
    print(f"  {overall_best['model']}  λ={overall_best['lambda']}  "
          f"RMSE={overall_best['val_rmse']:.3f} mg/dL")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    results, stats = run_grid_search()
    df = save_results(results)
