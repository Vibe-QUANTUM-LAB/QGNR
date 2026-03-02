# QGNR — Quantum Graphon Neural Representation

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%E2%80%932.2-ee4c2c.svg)](https://pytorch.org/)
[![POT](https://img.shields.io/badge/POT-0.9.x-green.svg)](https://pythonot.github.io/)
[![License: MIT](https://img.shields.io/badge/license-MIT-lightgrey.svg)](LICENSE)

Implementation of the paper:

> **Unveiling the Nature of Graphs through Quantum Graphon Learning**
> *npj Quantum Information, 2025*

QGNR is a quantum machine learning extension of the [IGNR](https://github.com/Mishne-Lab/IGNR) framework.
It replaces the classical SIREN network with a **hybrid quantum–classical circuit** to learn a continuous
graphon representation from a collection of graphs of varying size, using Gromov–Wasserstein (GW) optimal
transport as the training signal.

---

## Features

| Feature | Description |
|---|---|
| **Quantum backbone** | Parameterised quantum circuit (PQC) via TorchQuantum: RZ/RY rotations + circular CNOT entanglement, trained end-to-end with PyTorch autograd |
| **Classical baseline** | Full SIREN (`SirenNet`) with sine / ReLU / sigmoid activations — switch with `--model IGNR` |
| **GW training signal** | Proximal-gradient Sinkhorn OT (`gwloss_pg`) used as a differentiable loss; warm-starting transport matrix across epochs |
| **Graphon evaluation** | GW distance between estimated graphon (resolution 1000) and 13 ground-truth graphon families |
| **Flexible architecture** | Hidden width and depth of the classical SIREN, and quantum hidden size / circuit depth, are all CLI-configurable |

---

## Installation

### 1. Clone and enter the repo

```bash
git clone https://github.com/Vibe-QUANTUM-LAB/QGNR.git
cd QGNR-main
```

### 2. Create an isolated environment (recommended)

```bash
conda create -n qgnr python=3.10
conda activate qgnr
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note on versions:** TorchQuantum 0.1.8 is only tested against PyTorch 2.1–2.2.
> Using a newer PyTorch (≥ 2.3) may cause internal `QuantumDevice` API errors.
> NumPy must stay on the 1.x series — both TorchQuantum and POT have compatibility issues with NumPy 2.x.

---

## Quick Start

All commands are run from the `QGNR/` subdirectory (where `train_GNR.py` lives):

```bash
cd QGNR-main/QGNR
```

### Train QGNR (quantum model)

```bash
python train_GNR.py \
    --model QGNR \
    --n-epoch 80 \
    --f-name qgnr_exp
```

### Train IGNR (classical SIREN baseline)

```bash
python train_GNR.py \
    --model IGNR \
    --mlp_dim_hidden 20,20,20 \
    --w0 30 \
    --n-epoch 80 \
    --f-name ignr_exp
```

### Full CLI reference

| Argument | Type | Default | Description |
|---|---|---|---|
| `--model` | `str` | `QGNR` | Model variant: `QGNR` (quantum) or `IGNR` (classical SIREN) |
| `--n-epoch` | `int` | `80` | Number of training epochs |
| `--f-sample` | `str` | `fixed` | Grid sampling: `fixed` (uniform) or `random` |
| `--w0` | `float` | `30` | Initial sine frequency (IGNR only) |
| `--mlp_dim_hidden` | `str` | `"20,20,20"` | Hidden layer sizes for SIREN, comma-separated (IGNR only) |
| `--f-name` | `str` | `res` | Sub-directory name under `Result/` for saving outputs |
| `--data-path` | `str` | `Data/graphs.pkl` | Path to the graphs pickle file |

Results (GW error mean / std per graphon type) are printed to stdout and the estimated
graphons are saved under `Result/<f-name>/`.

---

## Project Structure

```
QGNR-main/
├── requirements.txt          # Pinned dependency versions
├── README.md                 # This file
└── QGNR/
    ├── train_GNR.py          # Entry point — CLI args, experiment loop, result logging
    ├── model_GNR.py          # GNR / GNR_wrapper — model construction & training loop
    ├── siren_pytorch.py      # Neural network building blocks:
    │                         #   QuantumLayer  — parameterised quantum circuit (TorchQuantum)
    │                         #   HybridLayer   — classical linear + BN + quantum
    │                         #   Hybridren     — full hybrid network (QGNR backbone)
    │                         #   Siren / SirenNet — classical SIREN (IGNR backbone)
    ├── helper.py             # Utility functions:
    │                         #   proximal_ot / proximal_ot_torch  — Sinkhorn solver
    │                         #   node_cost_st / gw_cost           — GW cost matrices
    │                         #   gwloss_pg / gwloss_pg_torch      — differentiable GW loss
    │                         #   gw_distance / mse_sort           — evaluation metrics
    │                         #   synthesize_graphon               — 13 ground-truth graphon families
    │                         #   get_graphs                       — data loading helper
    └── Data/
        ├── graphs.zip        # Compressed graph dataset (extract before use)
        └── graphs.pkl        # Extracted graph dataset (after unzip)
```

### Module dependency diagram

```
train_GNR.py
    ├── model_GNR.py
    │       ├── helper.py          (OT/GW computation, evaluation, graphon synthesis)
    │       └── siren_pytorch.py   (QuantumLayer / Hybridren / SirenNet)
    └── helper.py                  (gw_distance — evaluation only)
```

---

## Acknowledgements

- Graphon simulation and proximal-gradient GW code adapted from:
  Xu et al., *"Learning graphons via structured Gromov-Wasserstein barycenters"*, AAAI 2021
  [[paper]](https://arxiv.org/abs/2012.12654) · [[code]](https://github.com/HongtengXu/SGWB-Graphon)

- Classical IGNR baseline:
  Xia et al., *"Implicit Graphon Neural Representation"*, AISTATS 2023
  [[paper]](https://proceedings.mlr.press/v206/xia23a.html) · [[code]](https://github.com/Mishne-Lab/IGNR)

- Quantum circuit simulation:
  Han et al., *TorchQuantum* [[repo]](https://github.com/mit-han-lab/torchquantum)
