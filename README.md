# PINNModel

A Physics-Informed Neural Network (PINN) for modeling glucose-insulin dynamics in Type 1 diabetes, built on the UVA/Padova T1D Simulator.

## What this is

Standard neural networks trained on glucose-insulin data tend to learn spurious correlations and fail outside their training distribution. PINNs fix this by encoding the known ODEs governing glucose-insulin physiology directly into the loss function — so the model can't just memorize trajectories, it has to respect the underlying dynamics.

This project trains a PINN on simulated patient data from the UVA/Padova T1D Simulator and compares it against a Neural ODE baseline. The goal is to get accurate, physically consistent predictions of glucose response to insulin inputs, which matters a lot for closed-loop insulin delivery systems.

## Background

The glucose-insulin system is well-described by a set of coupled ODEs (the Hovorka model / Bergman minimal model family). The UVA/Padova simulator is the FDA-accepted gold standard for in silico T1D trials — it generates realistic virtual patient data across meal scenarios and insulin regimens.

PINNs embed these dynamics as soft constraints via a physics residual loss term, penalizing predictions that violate the ODEs. This makes them particularly useful when measured data is sparse or noisy, which is exactly the clinical reality.

## Methods

- **Architecture**: Feedforward neural network with physics residual loss
- **Physics constraints**: ODE residuals from the glucose-insulin compartment model added to the training loss
- **Baseline**: Neural ODE (adjoint method) for comparison
- **Simulator**: UVA/Padova T1D Simulator for synthetic patient data generation
- **Framework**: PyTorch

Loss function:

```
L_total = L_data + lambda * L_physics
```

where `L_physics` is the mean squared ODE residual evaluated at collocation points.

## Repo structure

```
PINNModel/
├── src/          # model architecture, training loop, loss functions
├── docs/         # writeup / figures
```

## Setup

```bash
git clone https://github.com/preelshh/PINNModel.git
cd PINNModel
pip install -r requirements.txt
```

## Usage

```bash
# Train the PINN
python src/train.py

# Run evaluation
python src/evaluate.py
```

## Results

The PINN maintains physiologically plausible glucose trajectories even under distribution shift (unseen meal sizes, timing perturbations), while the Neural ODE baseline degrades more gracefully but without hard physics guarantees.

## References

- Raissi et al., "Physics-informed neural networks," *Journal of Computational Physics*, 2019
- Dalla Man et al., "The UVA/PADOVA Type 1 Diabetes Simulator," *Journal of Diabetes Science and Technology*, 2014
- Chen et al., "Neural Ordinary Differential Equations," NeurIPS 2018

## Authors

Built as part of a class project on computational modeling of T1D dynamics.
