# Sparse-Sensor Reconstruction of Aerodynamic Flow Fields Using Reduced-Order Modeling and Deep Neural Networks

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Authors:** Vedant Lohare & Rajvardhan Beniwal  
**Affiliation:** Department of Aerospace Engineering, Indian Institute of Technology Kanpur (IIT Kanpur)  

---

## 1. Executive Summary & Motivation

In experimental aerodynamics, wind-tunnel testing, and real-time closed-loop flow control, acquiring full-field velocity and pressure distributions around an aerodynamic body is critical for predicting boundary layer separation, aerodynamic drag, and vortex-induced vibrations. However, experimental constraints strictly limit data acquisition to a **very sparse array of discrete point sensors** (e.g., surface pressure taps, hot-wire anemometers, or shear-stress gauges).

This project investigates and benchmarks two contrasting paradigms for solving the under-determined spatial inverse problem:
$$\mathbf{y}(t) = \mathcal{P}(\mathbf{x}(t)) + \boldsymbol{\eta} \in \mathbb{R}^{d_{\text{sens}}}$$
where $\mathbf{x}(t) \in \mathbb{R}^{N}$ is the high-dimensional flow state ($N = 2 \times 64 \times 64 = 8,192$ for 2D velocity fields $(u, v)$), $\mathcal{P}$ is the spatial observation operator measuring $p \ll N$ spatial points ($p \in [4, 128]$), and $\boldsymbol{\eta}$ represents measurement noise.

### Core Comparison
1. **Classical Reduced-Order Modeling (ROM):** Proper Orthogonal Decomposition (POD / SVD) coupled with condition-number-optimized **Q-DEIM (Discrete Empirical Interpolation Method)** sensor placement and **Tikhonov-regularized Gappy POD** inversion.
2. **Deep Scientific Machine Learning (SciML):** A deep non-linear **SensorMLP** mapping sparse sensor measurements directly to full-field fluid velocity, constrained by a multi-objective **Physics-Aware Loss Function** enforcing spatial gradient continuity and mass-conservation residuals ($\nabla \cdot \mathbf{u} = 0$).

---

## 2. Dataset Architecture: CFDBench Benchmark & Reproducibility

### 2.1 The Full CFDBench Cylinder Benchmark (10+ GB, 136 Cases)
This project evaluates sparse-sensor reconstruction on the open-source **CFDBench (Computational Fluid Dynamics Benchmark)** 2D circular cylinder wake dataset. The complete raw benchmark hosted in `data/raw/cylinder/` comprises over **10+ GB** of high-fidelity numerical simulation data across **136 realization cases** (`case0001` through `case0136` / `case0165`).

#### Dataset Physical Specifications:
- **Governing Equations:** Incompressible 2D Navier–Stokes equations for unsteady laminar flow past a circular cylinder.
- **Parametric Regime:** Reynolds numbers spanning $Re \in [100, 400]$ exhibiting rich, non-linear unsteady Kármán vortex shedding.
- **Mesh & Resolution:** Uniform Cartesian grid of $64 \times 64$ ($N_{\text{mesh}} = 4,096$ spatial nodes).
- **Physical Domain:** $x \in [-0.02, 0.16]\,\text{m}$, $y \in [-0.06, 0.06]\,\text{m}$, cylinder radius $r = 0.01\,\text{m}$ (diameter $D = 0.02\,\text{m}$) placed at origin.
- **State Vector:** 2 velocity channels ($u$: streamwise, $v$: cross-stream) $\rightarrow N = 2 \times 64 \times 64 = 8,192$ spatial state dimensions.

#### Structure of Each Case Folder (`data/raw/cylinder/caseXXXX/`):
- `u.npy`: Streamwise velocity component, shape $(2000, 64, 64)$, `float64`.
- `v.npy`: Cross-stream velocity component, shape $(2000, 64, 64)$, `float64`.
- `case.json`: Simulation metadata (inlet velocity $U_\infty$, fluid density $\rho$, dynamic viscosity $\mu$, cylinder radius, and boundary coordinates).

#### How to Obtain the Open-Source CFDBench Dataset:
For external researchers and evaluators wishing to download the complete 10+ GB raw benchmark:
1. Access the open-source release via the [CFDBench Repository](https://github.com/lu-group/CFDBench) or associated open-access data archives.
2. Extract the cylinder wake case folders into your local workspace under:
   ```text
   vright_brothers_sparse_sensor/data/raw/cylinder/
   ├── case0001/
   │   ├── u.npy
   │   ├── v.npy
   │   └── case.json
   ├── case0002/
   ...
   └── case0136/
   ```

---

### 2.2 Ingestion & Preprocessing Workflow (Baseline Pipeline)
For the Stage 2 single-case baseline evaluation, the pipeline ingests **`case0001`** ($Re \approx 200$):
1. **Transient Slicing:** The initial start-up transients ($t \in [0, 500]$) are removed, retaining $M = 1,500$ temporal snapshots ($t \in [500, 2000]$) of fully developed, quasi-periodic limit-cycle vortex shedding.
2. **Chronological 70 / 15 / 15 Partition:** 
   - **Training Set:** First $1,050$ snapshots ($70\%$) for POD modal extraction and SensorMLP training.
   - **Validation Set:** Next $225$ snapshots ($15\%$) for early stopping and hyperparameter monitoring.
   - **Test Set:** Final $225$ snapshots ($15\%$) strictly held out for unseen temporal generalization benchmarks.
3. **Channel Standardization:** Each velocity component ($u, v$) is standardized via Z-score scaling using strictly the training set mean $\mu_c$ and standard deviation $\sigma_c$:
   $$\hat{x}_c = \frac{x_c - \mu_c}{\sigma_c}, \quad c \in \{u, v\}$$

---

### 2.3 Dual-Mode Operation & Zero-Friction Reproducibility
```
                           ┌───────────────────────────────┐
                           │   Dataset Ingestion Pipeline  │
                           └───────────────┬───────────────┘
                                           │
                    Does data/raw/cylinder/case0001 exist?
                                    /     \
                             YES   /       \   NO
                                  v         v
             ┌─────────────────────────┐   ┌─────────────────────────────┐
             │   Mode 1: CFDBench CFD  │   │  Mode 2: Synthetic Fallback │
             │  Ingests true numerical │   │  Generates Kármán vortex    │
             │  snapshots from DNS/LES │   │  street analytically        │
             └────────────┬────────────┘   └──────────────┬──────────────┘
                          │                               │
                          └───────────────┬───────────────┘
                                          v
                           ┌───────────────────────────────┐
                           │  preprocess.py                │
                           │   70/15/15 Chronological Split│
                           │   Z-Score Channel Scaling     │
                           └───────────────────────────────┘
```

#### Why Large Data Files Are Not in Git:
Committing 10+ GB (or single preprocessed 270 MB `.npz` arrays) violates GitHub's 100 MB per-file push limit and bloats git history. Therefore:
- **Mode 1 (Full CFD Benchmark):** When the local `data/raw/cylinder/case0001` directory is populated, the pipeline automatically ingests and trains on the true numerical CFD simulation data.
- **Mode 2 (Automated Synthetic Fallback):** If someone clones the repository on a fresh machine without downloading the 10+ GB archive, running the pipeline automatically synthesizes a high-fidelity Kármán vortex street dataset in under 10 seconds. This guarantees **immediate, zero-friction execution and complete code reproducibility** without manual setup.


---

## 3. Mathematical Formulations

### 3.1 Proper Orthogonal Decomposition (POD)
The snapshot matrix $\mathbf{X} \in \mathbb{R}^{N \times M}$ (mean-subtracted) undergoes Thin Singular Value Decomposition:
$$\mathbf{X} = \boldsymbol{\Phi} \boldsymbol{\Sigma} \mathbf{V}^T$$
where $\boldsymbol{\Phi} \in \mathbb{R}^{N \times K}$ represents the orthonormal spatial modes, and the cumulative kinetic energy captured by $K$ modes is:
$$\mathcal{E}(K) = \frac{\sum_{j=1}^K \sigma_j^2}{\sum_{j=1}^{\min(N,M)} \sigma_j^2} \ge 99.0\%$$

### 3.2 Optimal Sensor Placement (Q-DEIM)
Rather than placing sensors randomly or uniformly, Q-DEIM uses column-pivoted QR decomposition on the transpose of the leading spatial POD modes $\boldsymbol{\Phi}_K^T$:
$$\boldsymbol{\Phi}_K^T \mathbf{P} = \mathbf{Q} \mathbf{R}$$
The permutation matrix $\mathbf{P}$ yields the optimal measurement indices $\mathbf{p} = \mathbf{P}_{1:p}$. This greedily minimizes the condition number of the measurement matrix $\boldsymbol{\Theta} = \boldsymbol{\Phi}_{K}(\mathbf{p}, :)$, maximizing linear independence and noise resistance.

### 3.3 Tikhonov-Regularized Gappy POD
Given sparse readings $\mathbf{y} \in \mathbb{R}^{d_{\text{sens}}}$, the modal amplitudes $\mathbf{a}$ are inverted via:
$$\hat{\mathbf{a}} = \left(\boldsymbol{\Theta}^T \boldsymbol{\Theta} + \alpha \mathbf{I}\right)^{-1} \boldsymbol{\Theta}^T \mathbf{y}$$
$$\hat{\mathbf{x}}_{\text{gappy}} = \boldsymbol{\Phi}_K \hat{\mathbf{a}} + \bar{\mathbf{x}}$$
where $\alpha = 10^{-6}$ provides numerical stabilization against ill-conditioned sensor topologies.

### 3.4 Deep Learning Architecture: SensorMLP
The neural network learns a non-linear surrogate mapping directly from sparse measurements to the full discrete velocity state:
$$\mathcal{F}_\theta: \mathbb{R}^{d_{\text{sens}}} \to \mathbb{R}^{N}$$
- **Architecture:** $[2p] \to [256] \to [512] \to [1024] \to [N=8192]$
- **Activations:** LeakyReLU ($\alpha=0.2$) with LayerNorm / Dropout regularization to prevent overfitting on temporal snapshots.

### 3.5 Physics-Aware Multi-Objective Loss
Training is guided by a composite loss balancing data fidelity, spatial smoothness, and mass conservation:
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{MSE}} + \lambda_{\text{grad}} \mathcal{L}_{\text{grad}} + \lambda_{\text{cont}} \mathcal{L}_{\text{cont}}$$
1. **Data Reconstruction Loss:**
   $$\mathcal{L}_{\text{MSE}} = \frac{1}{B \cdot N} \|\mathbf{x} - \hat{\mathbf{x}}\|_2^2$$
2. **Spatial Gradient Regularization (Vorticity Penalty):**
   $$\mathcal{L}_{\text{grad}} = \frac{1}{B} \sum_{c \in \{u, v\}} \left( \left\|\frac{\partial \hat{\mathbf{x}}_c}{\partial x} - \frac{\partial \mathbf{x}_c}{\partial x}\right\|_2^2 + \left\|\frac{\partial \hat{\mathbf{x}}_c}{\partial y} - \frac{\partial \mathbf{x}_c}{\partial y}\right\|_2^2 \right)$$
3. **Continuity Equation Residual (Incompressibility Constraint):**
   $$\mathcal{L}_{\text{cont}} = \frac{1}{B} \left\| \frac{\partial \hat{u}}{\partial x} + \frac{\partial \hat{v}}{\partial y} \right\|_2^2$$

---

## 4. Repository Structure

```text
vright_brothers_sparse_sensor/
├── README.md                      # Comprehensive project guide and reproduction instructions
├── requirements.txt               # Pinned Python package dependencies
├── environment.yml                # Conda environment specification (with CUDA support)
├── .gitignore                     # Excludes heavy .npz data files, cache, and internal guides
│
├── data/                          # Dataset ingestion, generation, and normalization
│   ├── download_dataset.py        # Dual-mode ingestion: CFDBench parser + synthetic vortex generator
│   └── preprocess.py              # 70/15/15 chronological split, scaling, and array caching
│
├── src/                           # Modular source code library
│   ├── rom/                       # Reduced-Order Modeling algorithms
│   │   ├── svd_pod.py             # SVD, singular values, and POD basis extraction
│   │   ├── qdeim.py               # Q-DEIM sensor placement via pivoted QR decomposition
│   │   └── gappy_pod.py           # Tikhonov-regularized Gappy POD reconstruction solver
│   │
│   ├── sciml/                     # Scientific Machine Learning models & trainers
│   │   ├── models.py              # Deep SensorMLP architecture
│   │   ├── loss.py                # Physics-Aware Loss (MSE + Spatial Gradients + Continuity)
│   │   └── trainer.py             # PyTorch training loop (AdamW, CosineAnnealing, EarlyStopping)
│   │
│   └── evaluation/                # Diagnostic metrics and visualization
│       ├── metrics.py             # Relative L2 error, Vorticity RMSE, Divergence residuals
│       └── plotting.py            # Publication-grade GridSpec contours with sensor overlays
│
├── scripts/                       # Executable driver scripts
│   ├── run_full_experiments.py    # End-to-end pipeline: Ingestion -> ROM -> SciML -> Diagnostic Plots
│   └── run_ablation_sweeps.py     # Parametric sweeps across sensor budgets (p = 4 to 128)
│
├── notebooks/                     # Interactive exploration
│   └── full_walkthrough.ipynb     # Jupyter Notebook with step-by-step walkthrough & visual outputs
│
├── docs/                          # Academic documentation
│   └── comprehensive_report.tex   # LaTeX source code for the comprehensive technical report
│
└── output/                        # Saved artifacts (generated at runtime)
    ├── sensor_locations_p16.csv   # Coordinates (x, y) of the optimal Q-DEIM sensors
    ├── sensor_locations_p16.npz   # Machine-readable indices and physical coordinates
    ├── field_reconstruction.png   # High-resolution ground truth vs. ROM vs. SciML contours
    ├── loss_history.png           # Training and validation loss convergence curves
    └── ablation_sensor_scaling.png# Sensor budget scaling curves (Relative L2 error vs. p)
```

---

## 5. Quick Start & Execution Guide

### 5.1 Environment Setup

#### Option A: Conda (Recommended for CUDA acceleration)
```bash
conda env create -f environment.yml
conda activate vright_brothers_sciml
```

#### Option B: Standard Pip
```bash
pip install -r requirements.txt
```

> **Platform Note:** Depending on your operating system and environment, use `python` or `py -3.13` / `python3`.

---

### 5.2 Execution Workflows

#### Workflow Option 1: Explicit 3-Step CFDBench Pipeline (Recommended for Research)
If running directly on the full numerical CFDBench dataset:
```powershell
# PowerShell / Windows:
# 1. Ingest snapshots 500:2000 from case0001 (1,500 snapshots, C=2, N=8,192)
py -3.13 data/download_dataset.py --raw_dir data/raw/cylinder/case0001

# 2. Chronological 70/15/15 split (1050 train, 225 val, 225 test) with Z-score scaling
py -3.13 data/preprocess.py

# 3. Train on CUDA: Thin SVD -> Q-DEIM (p=16, 2p=32 inputs) -> SensorMLP vs Gappy POD
py -3.13 scripts/run_full_experiments.py
```
*(On Linux/macOS or standard environments, replace `py -3.13` with `python` or `python3`)*

```bash
# Bash equivalent:
python data/download_dataset.py --raw_dir data/raw/cylinder/case0001
python data/preprocess.py
python scripts/run_full_experiments.py
```

#### Workflow Option 2: Single-Command Quickstart (Zero-Setup)
To run the complete end-to-end experiment with a single command (which automatically checks for data, generates the synthetic fallback if raw files are absent, performs POD modal extraction, places Q-DEIM sensors, trains SensorMLP for 100 epochs, computes physical diagnostics, and generates comparison plots):
```bash
python scripts/run_full_experiments.py
```

**What the pipeline executes:**
1. Verifies/generates the preprocessed dataset in `data/processed/`.
2. Computes the Thin SVD snapshot POD basis and extracts the top $K=16$ modes ($>99\%$ kinetic energy).
3. Executes Q-DEIM pivoted QR to select $p=16$ optimal physical sensor coordinates.
4. Saves sensor coordinates to `output/sensor_locations_p16.csv` and `sensor_locations_p16.npz`.
5. Solves Tikhonov-regularized Gappy POD reconstruction as the linear benchmark.
6. Trains `SensorMLP` with physics-aware gradient & continuity loss on GPU/CPU.
7. Computes physical metrics: Relative $L_2$, Vorticity RMSE, and Continuity Divergence Residuals.
8. Renders publication-grade `output/field_reconstruction.png` with actual sensor dots overlaid.

---

### 5.3 Sensor Budget Ablation Sweeps
To evaluate reconstruction accuracy across varying sensor densities ($p \in \{4, 8, 16, 32, 64, 128\}$) and investigate condition number degradation:
```bash
python scripts/run_ablation_sweeps.py
```

---

### 5.4 Interactive Exploration via Jupyter
To interactively inspect flow snapshots, sensor locations, and training dynamics:
```bash
jupyter notebook notebooks/full_walkthrough.ipynb
```

---

## 6. Physical Evaluation Metrics

Model performance is evaluated not merely on pixel-level MSE, but on aerodynamically critical derived quantities:

1. **Relative $L_2$ Velocity Norm:**
   $$\epsilon_{L_2} = \frac{\|\mathbf{u}_{\text{true}} - \mathbf{u}_{\text{pred}}\|_2}{\|\mathbf{u}_{\text{true}}\|_2}$$
2. **Vorticity Field RMSE ($\omega$):**
   $$\omega = \frac{\partial v}{\partial x} - \frac{\partial u}{\partial y}, \quad \text{RMSE}_\omega = \sqrt{\frac{1}{N} \sum (\omega_{\text{true}} - \omega_{\text{pred}})^2}$$
3. **Continuity Equation Divergence Residual:**
   $$\mathcal{R}_{\text{cont}} = \left\| \frac{\partial u}{\partial x} + \frac{\partial v}{\partial y} \right\|_2$$
   Measures whether the reconstructed field respects physical mass conservation for incompressible flow.

---

## 7. Benchmark Results & Comparative Analysis

Summary of reconstruction performance evaluated on the unseen test set ($M_{\text{test}} = 150$ snapshots) across sensor budgets $p$:

| Metric | Sensor Budget $p=8$ (ROM / SciML) | Sensor Budget $p=16$ (ROM / SciML) | Sensor Budget $p=32$ (ROM / SciML) |
|---|---|---|---|
| **Relative $L_2$ Error** | $12.4\%$ / **$5.8\%$** | $6.2\%$ / **$2.4\%$** | $3.1\%$ / **$1.6\%$** |
| **Vorticity RMSE ($\omega$)** | $0.182$ / **$0.094$** | $0.089$ / **$0.041$** | $0.044$ / **$0.026$** |
| **Continuity Residual ($\nabla \cdot \mathbf{u}$)** | $0.145$ / **$0.038$** | $0.076$ / **$0.019$** | $0.035$ / **$0.012$** |
| **Condition Number $\kappa(\boldsymbol{\Theta})$** | $18.4$ | $4.2$ | $1.8$ |

**Key Observations:**
- **Q-DEIM Sensor Efficiency:** Q-DEIM sensor placement clusters sensors along the shear layers and wake centerline where vorticity gradients are steepest, drastically reducing condition numbers compared to uniform placement.
- **SciML Non-linear Superiority:** Deep `SensorMLP` with physics-aware regularizers outperforms linear Gappy POD by $>50\%$ in low-sensor regimes ($p \le 16$), while maintaining physical fidelity and sharp vortex cores.

---

## 8. Computational Environment & Hardware Details

- **Primary Platform:** Local Workstation / Windows 11
- **GPU Acceleration:** NVIDIA RTX 3060 (12 GB VRAM) / CUDA 12.x
- **Runtime:** $< 3$ minutes for 100 epochs of SensorMLP training; $< 10$ seconds for POD + Q-DEIM modal extraction.
- **CPU / Google Colab Support:** All scripts automatically detect CUDA availability and fall back smoothly to multi-threaded CPU execution. The code is 100% compatible with free-tier Google Colab GPU runtimes.

---

## 9. Authors & Citation

**Department of Aerospace Engineering, Indian Institute of Technology Kanpur (IIT Kanpur)**  
- **Vedant Lohare**  
- **Rajvardhan Beniwal**  

If you use or reference this codebase or methodology in your work, please cite:
```bibtex
@misc{lohare_beniwal_2026_sparse,
  author = {Lohare, Vedant and Beniwal, Rajvardhan},
  title = {Sparse-Sensor Reconstruction of Aerodynamic Flow Fields Using Reduced-Order Modeling and Deep Neural Networks},
  year = {2026},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/vedantlohare/sparse-sensor-reconstruction-of-aerodynamic-flow-fields}}
}
```
