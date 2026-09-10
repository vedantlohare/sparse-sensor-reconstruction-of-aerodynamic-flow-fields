# Sparse-Sensor Reconstruction of Aerodynamic Flow Fields Using Reduced-Order Modeling and Deep Neural Networks

**Team:** VRIGHT BROTHERS
**Members:** Vedant Lohare & Rajvardhan Beniwal
**Affiliation:** Department of Aerospace Engineering, Indian Institute of Technology Kanpur (IIT Kanpur)
**Course:** AE646 Scientific Machine Learning for Fluid Mechanics

## Overview
This repository implements the baseline mathematical frameworks (Gappy POD with Q-DEIM sensor placement) and advanced nonlinear deep learning architectures (Physics-Aware Sensor-to-Field MLPs and DeepONets) to reconstruct high-fidelity, high-dimensional fluid dynamics fields from extremely sparse point-sensor measurements.

## Repository Structure
```text
vright_brothers_sparse_sensor/
├── data/                  # Automated dataset download and preprocessing
├── src/                   # Core implementation modules
│   ├── rom/               # SVD, POD, Gappy POD, Q-DEIM
│   ├── sciml/             # PyTorch Models, Physics-Aware Loss, Trainer
│   └── evaluation/        # Physical metrics (Vorticity, Continuity) and GridSpec plots
├── scripts/               # End-to-end execution scripts
├── notebooks/             # Interactive Jupyter walkthroughs
└── docs/                  # LaTeX project report and PDF
```

## Quick Start & Reproducibility

### 1. Environment Setup
Create the Conda environment to ensure identical PyTorch, CUDA, and numerical library versions:
```bash
conda env create -f environment.yml
conda activate vright_brothers_sciml
```
Or use pip:
```bash
pip install -r requirements.txt
```

### 2. Generate Data and Run Full Experiments
To automatically generate the synthetic dataset, partition it, perform modal extraction, place optimal sensors, train the deep neural networks, and generate the final output plots, run the main script:
```bash
python scripts/run_full_experiments.py
```

### 3. Run Ablation Sweeps
To test the sensor budget scaling laws and condition number degradation (e.g., $p \in [4, 128]$):
```bash
python scripts/run_ablation_sweeps.py
```

## Computational Environment & Hardware Requirements
**Computational Platform Used:** Local Setup / Windows Workstation
- **GPU:** NVIDIA CUDA-compatible GPU (e.g., RTX 3060)
- **Frameworks:** PyTorch 2.0+, NumPy, SciPy

All scripts have been configured to automatically detect and utilize an NVIDIA CUDA GPU if available. On a standard local workstation GPU or Google Colab, the full training loop takes `< 5` minutes per iteration.

*(Note: The code is fully runnable locally or can be easily uploaded and executed within Google Colab by copying the directory and running the main script.)*
