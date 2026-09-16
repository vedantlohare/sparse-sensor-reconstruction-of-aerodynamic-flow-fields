# VRIGHT BROTHERS: Sparse-Sensor Reconstruction of Aerodynamic Flow Fields
## Comprehensive Project Guide & Technical Handbook

**Authors:** Vedant Lohare (241151) & Rajvardhan Beniwal (240838)  
**Affiliation:** Department of Aerospace Engineering, Indian Institute of Technology Kanpur  
**Course:** AE646: Scientific Machine Learning for Fluid Mechanics  
**Repository:** `vright_brothers_sparse_sensor/`

---

## Table of Contents
1. [Executive Summary & Motivation](#1-executive-summary--motivation)
2. [Physical Problem Formulation & Dataset Engineering](#2-physical-problem-formulation--dataset-engineering)
   - [2.1 Fluid Dynamics: 2D Cylinder Wake](#21-fluid-dynamics-2d-cylinder-wake)
   - [2.2 Ingestion of the CFDBench Dataset](#22-ingestion-of-the-cfdbench-dataset)
   - [2.3 Transient Slicing & Dimensionality](#23-transient-slicing--dimensionality)
   - [2.4 Chronological Splitting & Standardization](#24-chronological-splitting--standardization)
3. [Mathematical Foundations & Algorithmic Derivations](#3-mathematical-foundations--algorithmic-derivations)
   - [3.1 Proper Orthogonal Decomposition (POD / SVD)](#31-proper-orthogonal-decomposition-pod--svd)
   - [3.2 Optimal Sensor Placement via Q-DEIM](#32-optimal-sensor-placement-via-q-deim)
   - [3.3 Classical Reconstruction: Tikhonov-Regularized Gappy POD](#33-classical-reconstruction-tikhonov-regularized-gappy-pod)
   - [3.4 Deep Learning Architecture: SensorMLP](#34-deep-learning-architecture-sensormlp)
   - [3.5 Physics-Aware Loss Formulation](#35-physics-aware-loss-formulation)
   - [3.6 Optimization & Training Dynamics](#36-optimization--training-dynamics)
4. [Codebase Architecture & File-by-File Walkthrough](#4-codebase-architecture--file-by-file-walkthrough)
5. [Evaluation Metrics & Physical Diagnostics](#5-evaluation-metrics--physical-diagnostics)
6. [Step-by-Step Execution Guide](#6-step-by-step-execution-guide)
7. [Expected Results & Interpretation](#7-expected-results--interpretation)
8. [Troubleshooting & Frequently Asked Questions (FAQ)](#8-troubleshooting--frequently-asked-questions-faq)
9. [Stage 3 Roadmap & Future Extensions](#9-stage-3-roadmap--future-extensions)

---

## 1. Executive Summary & Motivation

In aerodynamic engineering and experimental fluid mechanics, obtaining full-field velocity and pressure data around an aerospace vehicle (such as a transonic wing, turbine blade, or bluff body) is critical for:
- Monitoring boundary layer transition and flow separation,
- Calculating lift, drag, and unsteady structural loading,
- Designing active closed-loop flow control systems.

However, high-fidelity numerical simulations (Direct Numerical Simulation - DNS, Large Eddy Simulation - LES) require billions of floating-point operations and cannot be solved in real time. Conversely, physical experimental facilities (wind tunnels, flight test aircraft) can only host a **very sparse array of sensors** (e.g., surface pressure taps, hot-wire anemometers, or shear stress sensors) due to structural, electrical, weight, and aerodynamic intrusion constraints.

### The Core Problem: The Under-Determined Spatial Inverse Problem
Let the true, high-resolution physical state vector of the fluid at time $t$ be:
$$\mathbf{x}(t) \in \mathbb{R}^N$$
where $N$ is very large (e.g., $N = 8,192$ or $10^6$ degrees of freedom).

We only measure the flow at $p$ discrete sensor locations:
$$\mathbf{y}(t) = \mathcal{P}(\mathbf{x}(t)) + \boldsymbol{\eta} \in \mathbb{R}^{d_{sens}}$$
where $d_{sens} \ll N$ (typically $p \sim 16$ sensors), $\mathcal{P}$ is a spatial measurement/selection operator, and $\boldsymbol{\eta} \sim \mathcal{N}(0, \sigma^2 \mathbf{I})$ represents measurement noise.

**The Goal of This Project:**
Develop and benchmark two distinct reconstruction paradigms to reconstruct the full state $\mathbf{x}$ from sparse readings $\mathbf{y}$:
1. **Classical Reduced-Order Modeling (ROM):** Proper Orthogonal Decomposition (POD) coupled with Q-DEIM sensor placement and Tikhonov-regularized Gappy POD.
2. **Scientific Machine Learning (SciML):** A Physics-Aware Multi-Layer Perceptron (`SensorMLP`) that directly maps sparse readings to the full flow field, penalizing non-physical gradients and mass-conservation violations.

---

## 2. Physical Problem Formulation & Dataset Engineering

### 2.1 Fluid Dynamics: 2D Cylinder Wake
The canonical benchmark examined is the **two-dimensional incompressible flow past a circular cylinder** at a laminar vortex-shedding Reynolds number ($Re \approx 100-150$). 

As the incoming fluid meets the cylinder:
1. Boundary layers separate on the upper and lower surfaces,
2. Unstable shear layers roll up into alternating vortices,
3. A periodic, oscillatory wake known as the **von Kármán vortex street** forms downstream, characterized by a non-dimensional Strouhal shedding frequency:
   $$St = \frac{f D}{U_\infty} \approx 0.2$$

### 2.2 Ingestion of the CFDBench Dataset
Our real aerodynamic data originates from the benchmark suite **CFDBench** (`data/raw/cylinder/case0001/`):
- `u.npy`: Streamwise velocity $u(t, y, x)$ across 2000 time snapshots on a $64 \times 64$ Cartesian grid.
- `v.npy`: Cross-stream velocity $v(t, y, x)$ across 2000 time snapshots on a $64 \times 64$ Cartesian grid.
- `case.json`: Simulation metadata and boundary conditions.

> [!NOTE]
> In CFDBench 2D cylinder simulations, the original physical state comprises **$C = 2$ channels** ($u$ and $v$) at a $64 \times 64$ resolution.
> To align with our Stage 1 research proposal, we:
> 1. Interpolate the raw fields to a higher-aspect **$64 \times 128$ grid** using bicubic interpolation.
> 2. Analytically derive the **pressure field $p$** by solving the 2D incompressible pressure-Poisson equation $\nabla^2 p = -\rho \nabla \cdot (\mathbf{u} \cdot \nabla \mathbf{u})$ via a 5-point discrete Laplacian. Since true pressure boundary conditions are not strictly provided in the raw dataset, we assume reference **Dirichlet boundary conditions ($p=0$)** along all computational domain boundaries. This is a standard assumption for derived ML fields to ensure a unique, stable solution.

### 2.3 Transient Slicing & Dimensionality
When fluid simulations begin from uniform flow, the flow exhibits an initial numerical and physical transient before settling into a stable periodic limit cycle.
- **Transients:** Snapshots $0 \le t < 500$
- **Selected Dataset:** Snapshots $500 \le t < 2000 \implies M = 1500$ fully developed vortex-shedding snapshots.

**State Dimensions:**
- Grid Size: $N_y = 64, N_x = 128$ ($8,192$ spatial coordinates)
- Channels: $C = 3$ ($u, v, p$)
- State Vector Size: $N = C \times N_y \times N_x = 3 \times 64 \times 128 = 24,576$ components per snapshot.
- Full Snapshot Matrix: $\mathbf{X} \in \mathbb{R}^{24576 \times 1500}$

### 2.4 Chronological Splitting & Standardization
In dynamical systems, shuffling data randomly causes **data leakage** because snapshot $t$ is strongly correlated with $t-1$ and $t+1$. We enforce a strict **chronological partition**:
- **Train Set (70%):** Snapshots $0 \dots 1049$ ($M_{train} = 1050$)
- **Validation Set (15%):** Snapshots $1050 \dots 1274$ ($M_{val} = 225$)
- **Test Set (15%):** Snapshots $1275 \dots 1499$ ($M_{test} = 225$)

**Z-Score Normalization:**
To prevent scale imbalances during neural network training, each velocity channel $c \in \{0, 1\}$ is standardized using statistics computed **strictly from the training set**:
$$u_{norm} = \frac{u - \mu_u}{\sigma_u}, \quad v_{norm} = \frac{v - \mu_v}{\sigma_v}$$
The validation and test sets are transformed using the exact same $(\mu, \sigma)$ pairs.

---

## 3. Mathematical Foundations & Algorithmic Derivations

### 3.1 Proper Orthogonal Decomposition (POD / SVD)
POD extracts an optimal, orthonormal spatial basis $\boldsymbol{\Phi} = [\boldsymbol{\phi}_1, \boldsymbol{\phi}_2, \dots, \boldsymbol{\phi}_r]$ that captures the maximum possible kinetic energy of the flow fluctuations.

1. **Mean Subtraction:**
   $$\bar{\mathbf{x}} = \frac{1}{M_{train}} \sum_{m=1}^{M_{train}} \mathbf{x}_m, \quad \tilde{\mathbf{X}} = [\mathbf{x}_1 - \bar{\mathbf{x}}, \dots, \mathbf{x}_{M_{train}} - \bar{\mathbf{x}}] \in \mathbb{R}^{N \times M_{train}}$$
2. **Thin Singular Value Decomposition (SVD):**
   $$\tilde{\mathbf{X}} = \boldsymbol{\Phi} \boldsymbol{\Sigma} \mathbf{V}^T$$
   where $\boldsymbol{\Phi} \in \mathbb{R}^{N \times M_{train}}$ contains the spatial POD modes, and $\boldsymbol{\Sigma} = \text{diag}(\sigma_1, \sigma_2, \dots)$ contains the singular values ordered such that $\sigma_1 \ge \sigma_2 \ge \dots \ge 0$.
3. **Modal Truncation Criterion:**
   We retain the first $r$ modes capturing at least 95% of the total fluctuation kinetic energy:
   $$\text{Cumulative Energy}(r) = \frac{\sum_{i=1}^r \sigma_i^2}{\sum_{i=1}^{M_{train}} \sigma_i^2} \ge 0.95$$
   For the periodic cylinder wake, the first 2 POD modes represent the primary vortex-shedding limit cycle (a harmonic conjugate pair), capturing over 80-90% of the energy alone!

### 3.2 Optimal Sensor Placement via Q-DEIM
Given a budget of $p$ sensors, where should we place them in the physical domain? Placing sensors arbitrarily or uniformly can cause severe matrix ill-conditioning.

**Q-DEIM (Discrete Empirical Interpolation Method with QR Pivoting):**
Q-DEIM applies column-pivoted QR decomposition to the transposed POD modes $\boldsymbol{\Phi}_r^T \in \mathbb{R}^{r \times N}$:
$$\boldsymbol{\Phi}_r^T \mathbf{P} = \mathbf{Q} \mathbf{R}$$
where:
- $\mathbf{P}$ is a permutation matrix (or vector of column indices),
- $\mathbf{Q}$ is an orthogonal matrix,
- $\mathbf{R}$ is an upper-triangular matrix whose diagonal elements $|R_{ii}|$ decay monotonically.

**Multi-Channel Spatial Energy Pooling:**
In physical fluid flow, a physical sensor probe placed at grid point $(x_i, y_i)$ measures **both** $u$ and $v$ simultaneously. Therefore, we pool the modal energy across channels:
$$\boldsymbol{\Phi}_{spatial}(k, j) = \sqrt{\sum_{c=0}^{C-1} \boldsymbol{\Phi}(k + c \cdot N_{spatial}, j)^2}, \quad k \in [0, N_{spatial}-1]$$
Performing pivoted QR on $\boldsymbol{\Phi}_{spatial}^T$ yields the top $p$ spatial grid coordinates:
$$\mathbf{s} = [s_1, s_2, \dots, s_p], \quad s_i \in [0, N_{spatial}-1]$$
Measuring $C=2$ velocity components at each of these $p$ locations yields an input vector of dimension:
$$d_{sens} = C \times p = 2p$$

### 3.3 Classical Reconstruction: Tikhonov-Regularized Gappy POD
In Gappy POD, the flow field fluctuation is approximated as a linear combination of the $r$ POD modes:
$$\tilde{\mathbf{x}} \approx \boldsymbol{\Phi}_r \mathbf{a}$$
At the $2p$ sensor locations, the sparse measurement vector is:
$$\mathbf{y}_s = \mathbf{C} \tilde{\mathbf{x}} = \mathbf{C} \boldsymbol{\Phi}_r \mathbf{a} = \boldsymbol{\Phi}_s \mathbf{a}$$
where $\boldsymbol{\Phi}_s = \mathbf{C} \boldsymbol{\Phi}_r \in \mathbb{R}^{2p \times r}$ is the sub-matrix of POD modes evaluated at the sensor indices.

When $2p < r$, the point-collocation system is under-determined and ill-conditioned. We solve for the modal coefficients $\hat{\mathbf{a}}$ using **Tikhonov Regularization**:
$$\hat{\mathbf{a}} = \arg\min_{\mathbf{a}} \left( \|\mathbf{y}_s - \boldsymbol{\Phi}_s \mathbf{a}\|_2^2 + \mu \|\mathbf{a}\|_2^2 \right)$$
Taking the gradient with respect to $\mathbf{a}$ and setting to zero:
$$\hat{\mathbf{a}} = \left( \boldsymbol{\Phi}_s^T \boldsymbol{\Phi}_s + \mu \mathbf{I}_r \right)^{-1} \boldsymbol{\Phi}_s^T \mathbf{y}_s$$
The full state is then reconstructed as:
$$\hat{\mathbf{x}} = \bar{\mathbf{x}} + \boldsymbol{\Phi}_r \hat{\mathbf{a}}$$

### 3.4 Deep Learning Architecture: SensorMLP
Classical Gappy POD is fundamentally limited because it projects onto a **linear subspace**. Real turbulent and separated flows evolve on complex, curved **nonlinear manifolds**.

To capture this nonlinearity, we design **`SensorMLP`** ($f_\theta: \mathbb{R}^{2p} \to \mathbb{R}^N$):
```
Sparse Input: y (dim = 2p)
   │
   ▼
[Linear(2p -> 128)  -> LayerNorm(128)  -> GELU()]
   │
   ▼
[Linear(128 -> 256) -> LayerNorm(256)  -> GELU()]
   │
   ▼
[Linear(256 -> 512) -> LayerNorm(512)  -> GELU()]
   │
   ▼
[Linear(512 -> 1024)-> LayerNorm(1024) -> GELU()]
   │
   ▼
[Linear(1024 -> N=24576)]
   │
   ▼
Full Reconstructed Field: x_hat (dim = 24576)
```

- **LayerNorm:** Stabilizes the internal covariate shift across layers.
- **GELU (Gaussian Error Linear Unit):** Provides smooth, non-monotonic curvature, outperforming standard ReLU in fluid flow field regression.

### 3.5 Physics-Aware Loss Formulation
A naive neural network trained purely with Mean Squared Error (MSE) tends to produce overly blurred flow fields that underestimate shear-layer velocity gradients and violate mass conservation.

We formulate a **composite Physics-Aware Loss**:
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{MSE}} + \gamma_1 \mathcal{L}_{\text{grad}} + \gamma_2 \mathcal{L}_{\text{div}}$$

1. **State MSE Loss:**
   $$\mathcal{L}_{\text{MSE}} = \frac{1}{N} \|\hat{\mathbf{x}} - \mathbf{x}_{\text{true}}\|_2^2$$
2. **Sobolev / Spatial Gradient Regularizer:**
   Using central finite differences (`torch.gradient`), we compute the spatial derivatives $\nabla_x$ and $\nabla_y$ of the reconstructed and ground-truth 2D fields:
   $$\mathcal{L}_{\text{grad}} = \|\nabla_x \hat{\mathbf{x}} - \nabla_x \mathbf{x}_{\text{true}}\|_1 + \|\nabla_y \hat{\mathbf{x}} - \nabla_y \mathbf{x}_{\text{true}}\|_1$$
   This penalizes spatial smoothing and preserves sharp vortex cores.
3. **Continuity / Divergence Residual:**
   For an incompressible fluid, the continuity equation requires zero divergence:
   $$\nabla \cdot \vec{u} = \frac{\partial u}{\partial x} + \frac{\partial v}{\partial y} = 0$$
   $$\mathcal{L}_{\text{div}} = \frac{1}{N_{\text{spatial}}} \sum_{i, j} \left| \frac{\partial \hat{u}}{\partial x} + \frac{\partial \hat{v}}{\partial y} \right|$$

### 3.6 Optimization & Training Dynamics
- **Optimizer:** Adam ($\beta_1 = 0.9, \beta_2 = 0.999$, weight decay $= 10^{-4}$).
- **Base Learning Rate:** $\eta_0 = 10^{-3}$.
- **Learning Rate Scheduler:** `ReduceLROnPlateau(mode='min', factor=0.5, patience=5)`. When validation loss stops improving for 5 epochs, the learning rate drops by 50%.
- **Training Budget:** 100 epochs.
- **Early Stopping:** `patience = 15`. If validation loss does not reach a new minimum for 15 consecutive epochs, training terminates and the best checkpoint weights are restored.

---

## 4. Codebase Architecture & File-by-File Walkthrough

The repository is structured following clean scientific software engineering principles:

```
vright_brothers_sparse_sensor/
├── data/
│   ├── download_dataset.py     # Parses raw CFDBench .npy or generates synthetic wake
│   └── preprocess.py           # Z-score standardization and chronological train/val/test split
├── src/
│   ├── rom/
│   │   ├── svd_pod.py          # SVD extraction, fluctuation centering, cumulative energy
│   │   ├── qdeim.py            # Pivoted QR decomposition for optimal multi-channel sensor selection
│   │   └── gappy_pod.py        # Tikhonov-regularized least squares reconstruction
│   ├── sciml/
│   │   ├── models.py           # PyTorch SensorMLP and DeepONet architectures
│   │   ├── loss.py             # PhysicsAwareLoss with torch.gradient finite differences
│   │   └── trainer.py          # CUDA-accelerated training loop with scheduler & early stopping
│   └── evaluation/
│       ├── metrics.py          # Relative L2, 2D vorticity fidelity, continuity divergence
│       └── plotting.py         # 4-panel publication-grade contour comparison figures
├── scripts/
│   ├── run_full_experiments.py # Primary end-to-end benchmark script (100 epochs)
│   └── run_ablation_sweeps.py  # Sensor budget scaling sweep (p in [4, 8, 16, 32, 64, 128])
├── docs/
│   ├── PROJECT_GUIDE.md        # Comprehensive technical guide (this document)
│   └── comprehensive_report.tex# LaTeX interim project report
├── environment.yml             # Conda environment specification
├── requirements.txt            # Pip dependencies
└── README.md                   # Quickstart instructions
```

### Detailed File Responsibilities:
1. **[data/download_dataset.py](file:///c:/STUDY/PROJECTS/sparse-sensor-reconstruction-of-aerodynamic-flow-fields/vright_brothers_sparse_sensor/data/download_dataset.py)**:
   - Checks `--raw_dir` for `u.npy` and `v.npy`.
   - Slices snapshots 500 to 2000 (discarding startup transients).
   - Stacks them into shape `(1500, 2, 64, 64)`.
   - Fallback: If raw CFDBench files are absent, an analytical Biot-Savart vortex street model generates high-quality synthetic wake data automatically.
2. **[data/preprocess.py](file:///c:/STUDY/PROJECTS/sparse-sensor-reconstruction-of-aerodynamic-flow-fields/vright_brothers_sparse_sensor/data/preprocess.py)**:
   - Partitions data chronologically: 1050 train, 225 val, 225 test snapshots.
   - Computes channel-wise $\mu$ and $\sigma$ strictly on train split.
   - Outputs compressed archives: `train.npz`, `val.npz`, and `test.npz`.
3. **[src/rom/svd_pod.py](file:///c:/STUDY/PROJECTS/sparse-sensor-reconstruction-of-aerodynamic-flow-fields/vright_brothers_sparse_sensor/src/rom/svd_pod.py)**:
   - Computes thin SVD using `scipy.linalg.svd(full_matrices=False)`.
   - Calculates cumulative singular value energy spectrum.
4. **[src/rom/qdeim.py](file:///c:/STUDY/PROJECTS/sparse-sensor-reconstruction-of-aerodynamic-flow-fields/vright_brothers_sparse_sensor/src/rom/qdeim.py)**:
   - Pools modal energy across the $C=2$ velocity channels.
   - Executes `scipy.linalg.qr(Phi_spatial.T, pivoting=True)`.
   - Returns the top $p$ spatial sensor coordinate indices.
5. **[src/rom/gappy_pod.py](file:///c:/STUDY/PROJECTS/sparse-sensor-reconstruction-of-aerodynamic-flow-fields/vright_brothers_sparse_sensor/src/rom/gappy_pod.py)**:
   - Builds point collocation matrix $\boldsymbol{\Phi}_s$.
   - Inverts $(\boldsymbol{\Phi}_s^T \boldsymbol{\Phi}_s + \mu \mathbf{I})$ to recover modal amplitudes $\hat{\mathbf{a}}$.
6. **[src/sciml/models.py](file:///c:/STUDY/PROJECTS/sparse-sensor-reconstruction-of-aerodynamic-flow-fields/vright_brothers_sparse_sensor/src/sciml/models.py)**:
   - Houses `SensorMLP` (dense nonlinear mapping) and `DeepONet` (operator learning trunk & branch networks).
7. **[src/sciml/loss.py](file:///c:/STUDY/PROJECTS/sparse-sensor-reconstruction-of-aerodynamic-flow-fields/vright_brothers_sparse_sensor/src/sciml/loss.py)**:
   - Reshapes flattened 1D network output back to 2D image tensors.
   - Uses `torch.gradient` to backpropagate directly through spatial finite-difference derivatives.
8. **[src/sciml/trainer.py](file:///c:/STUDY/PROJECTS/sparse-sensor-reconstruction-of-aerodynamic-flow-fields/vright_brothers_sparse_sensor/src/sciml/trainer.py)**:
   - Detects CUDA GPU (`torch.cuda.is_available()`).
   - Implements mini-batch training with gradient descent, LR plateau reduction, and validation early stopping.
9. **[scripts/run_full_experiments.py](file:///c:/STUDY/PROJECTS/sparse-sensor-reconstruction-of-aerodynamic-flow-fields/vright_brothers_sparse_sensor/scripts/run_full_experiments.py)**:
   - Executes the complete workflow end-to-end: POD $\to$ Q-DEIM $\to$ Gappy POD $\to$ SensorMLP $\to$ Metrics $\to$ Publication plot.
10. **[scripts/run_ablation_sweeps.py](file:///c:/STUDY/PROJECTS/sparse-sensor-reconstruction-of-aerodynamic-flow-fields/vright_brothers_sparse_sensor/scripts/run_ablation_sweeps.py)**:
    - Loops through sensor counts $p \in \{4, 8, 16, 32, 64, 128\}$.
    - Evaluates the scaling law of reconstruction error vs. sensor budget for both Gappy POD and SensorMLP.

---

## 5. Evaluation Metrics & Physical Diagnostics

To rigorously benchmark performance, we evaluate three complementary metrics on the unseen **Test Set**:

### 1. Global Relative $L_2$ Error Norm ($\%$)
Measures the overall state discrepancy:
$$\mathcal{E}_{L2} = \frac{1}{M_{test}} \sum_{m=1}^{M_{test}} \frac{\|\mathbf{x}_{\text{true}}^{(m)} - \hat{\mathbf{x}}^{(m)}\|_2}{\|\mathbf{x}_{\text{true}}^{(m)}\|_2} \times 100\%$$

### 2. Vorticity Fidelity Error ($\%$)
Vorticity $\omega_z = \frac{\partial v}{\partial x} - \frac{\partial u}{\partial y}$ characterizes coherent vortex structures, rotation, and shear layer dynamics:
$$\mathcal{E}_{\omega} = \frac{\|\boldsymbol{\omega}_{\text{true}} - \hat{\boldsymbol{\omega}}\|_2}{\|\boldsymbol{\omega}_{\text{true}}\|_2} \times 100\%$$
A model that achieves low $L_2$ error might still fail on vorticity if its spatial derivatives are poorly represented.

### 3. Mean Continuity Residual ($\bar{R}_{\text{div}}$)
Tests adherence to the mass-conservation governing law:
$$\bar{R}_{\text{div}} = \frac{1}{N_{\text{spatial}}} \sum_{i, j} \left| \frac{\partial \hat{u}}{\partial x} + \frac{\partial \hat{v}}{\partial y} \right|$$
Lower values indicate greater physical consistency.

---

## 6. Step-by-Step Execution Guide

### Prerequisites
A working Python environment (Python 3.10 to 3.13) with PyTorch and standard scientific packages.

```powershell
# Navigate to the project directory
cd c:\STUDY\PROJECTS\sparse-sensor-reconstruction-of-aerodynamic-flow-fields\vright_brothers_sparse_sensor

# If using pip in your current environment:
pip install -r requirements.txt
```

> [!NOTE]
> **Python Interpreter Command Variations Across Systems:**  
> The code examples below use `py -3.13` (the Windows Python Launcher targeting Python 3.13 on the author's development machine). Depending on your OS, desktop setup, or virtual environment, replace `py -3.13` with your environment's appropriate interpreter command:
> - **Active Conda / Virtualenv / Default Windows:** `python <script>.py`
> - **Linux / macOS / WSL:** `python3 <script>.py`
> - **Windows Python Launcher (default version):** `py <script>.py`
> - **Direct Conda Environment Invocation:** `conda run -n vright_sciml python <script>.py`

### Step 1: Ingest the Raw Data
Parse the CFDBench cylinder case (or automatically fall back to the synthetic generator if raw files are absent):
```powershell
# Windows (Python 3.13 Launcher)
py -3.13 data/download_dataset.py --raw_dir data/raw/cylinder/case0001
# Or standard cross-platform:
# python data/download_dataset.py --raw_dir data/raw/cylinder/case0001
```

### Step 2: Preprocess and Standardize
Compute normalization statistics and split into train, val, and test partitions:
```powershell
py -3.13 data/preprocess.py
# Or: python data/preprocess.py
```

### Step 3: Run the Full Benchmark Pipeline
Executes POD, Q-DEIM, Gappy POD, and trains `SensorMLP` for **100 epochs** on GPU:
```powershell
py -3.13 scripts/run_full_experiments.py --case case0001
# Or: python scripts/run_full_experiments.py --case case0001
```
*(You can target any of the 136 real dataset folders by passing `--case caseXXXX`).*
**What this produces:**
- Formatted Markdown performance comparison table in the terminal.
- Reconstructed flow field figure with Q-DEIM sensor overlay saved to `output/reconstruction_panel_test_10.png`.
- Optimal sensor locations saved to `output/optimal_sensors.csv` and `output/optimal_sensors.npz`.

### Step 4: Run Sensor Budget Ablations (Optional but Recommended)
Evaluates how accuracy scales across $p \in \{4, 8, 16, 32, 64, 128\}$:
```powershell
py -3.13 scripts/run_ablation_sweeps.py
# Or: python scripts/run_ablation_sweeps.py
```

### Step 5: Interactive Exploration via Jupyter Notebook
For a highly visual, step-by-step introduction to the mathematical foundations of this project (POD, Q-DEIM, and Gappy POD), we provide an interactive walkthrough notebook. This is perfect for professors, TAs, or researchers who want to inspect the math without parsing raw `.py` scripts.
```powershell
jupyter notebook notebooks/full_walkthrough.ipynb
```
**The notebook interactively demonstrates:**
- Loading and visualizing the flow field tensors.
- Computing the SVD and plotting the cumulative energy spectrum.
- Executing the Q-DEIM algorithm and overlaying exact physical sensor locations onto vorticity contours.
- Reconstructing the flow using Gappy POD and computing baseline error.

---

## 7. Expected Results & Interpretation

When running the pipeline on the test dataset ($p = 16$ sensors, capturing $2 \times 16 = 32$ velocity readings):

| $p$ | Method | Rel. $L_2$ Error | Vorticity Error | Mean Continuity Residual $\bar{R}_{\text{div}}$ |
|---|---|---|---|---|
| **16** | **Gappy POD (Linear ROM)** | 44.06% | 48.78% | 0.092 |
| **16** | **SensorMLP (Deep SciML)** | **3.09%** | **4.86%** | 0.163 |

### Why Does SensorMLP Outperform Gappy POD?
1. **Nonlinear Manifold Mapping:** Gappy POD forces the solution onto a linear hyperplane spanned by $\boldsymbol{\Phi}_r$. When vortex positions shift nonlinearly, linear projection creates "ghosting" artifacts. SensorMLP smoothly interpolates across nonlinear phases.
2. **Gradient and Continuity Penalties:** The Sobolev regularizer in `PhysicsAwareLoss` explicitly guides the network weights to respect flow gradients, resulting in significantly sharper vortex cores.
3. **Sensor Placement Synergy:** Q-DEIM places sensors precisely where the POD modes have the highest variability (in the wake separation bubble and along the shear layer), maximizing the information density fed into the neural network.

---

## 8. Troubleshooting & Frequently Asked Questions (FAQ)

### Q1: Why 100 epochs instead of 50?
**A:** In scientific regression tasks, training for 100 epochs ensures the model transitions from coarse large-scale pattern matching to fine shear-layer fitting. With our learning rate decay (`factor=0.5, patience=5`) and early stopping (`patience=15`), the training will safely terminate if the model reaches its minimum validation loss earlier, preventing any wasted compute.

### Q2: How does the code choose between GPU and CPU?
**A:** In `src/sciml/trainer.py`, the device is automatically configured:
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```
If an NVIDIA GPU with CUDA is detected, it logs `Training on GPU: <GPU_NAME>`. If no GPU is available, it gracefully falls back to CPU.

### Q3: What happens if `data/raw/cylinder/case0001` is missing?
**A:** `data/download_dataset.py` contains an automatic fallback. If the specified directory or `.npy` files cannot be found, it generates a mathematically rigorous unsteady vortex-street dataset using an analytical Biot-Savart vortex model. This guarantees that anyone cloning the repository can run the entire pipeline immediately!

### Q4: How do 1D sensor indices map back to 2D physical coordinates?
**A:** Given a spatial index $s \in [0, 4095]$ on a $64 \times 64$ grid:
$$\text{row} = s // 64, \quad \text{col} = s \% 64$$
$$x_{coord} = X[\text{row}, \text{col}], \quad y_{coord} = Y[\text{row}, \text{col}]$$
This mapping is implemented in `src/evaluation/plotting.py` to scatter the red sensor dots onto the flow contour plots.

---

## 9. Stage 3 Roadmap & Future Extensions

As we advance into **Stage 3** of the project, the following modules and investigations will be integrated:
1. **Sensor Measurement Noise Robustness:**
   - Inject zero-mean Gaussian noise $\boldsymbol{\eta} \sim \mathcal{N}(0, \sigma^2)$ into test sensor readings at levels $\sigma \in \{1\%, 5\%, 10\%, 20\%\}$ to evaluate noise resilience.
2. **DeepONet & Operator Learning:**
   - Benchmark the implemented `DeepONet` architecture (branch network for sparse sensors, trunk network for continuous spatial query coordinates $(x, y)$) to enable grid-independent continuous reconstruction.
3. **Convolutional & Voronoi Spatial Decoders:**
   - Implement Voronoi tessellation preprocessing to project sparse scattered points onto a 2D image, followed by a U-Net / Fourier Neural Operator (FNO) spatial decoder.
4. **Generalization Across Flow Regimes:**
   - Test reconstruction transferability across varying Reynolds numbers ($Re = 100 \to 400$) and different bluff body geometries.
