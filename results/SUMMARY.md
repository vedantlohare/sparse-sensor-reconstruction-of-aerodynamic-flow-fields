# Stage 3 Comprehensive Empirical Results Summary
## Team VRIGHT BROTHERS — AE646 Scientific Machine Learning for Fluid Mechanics

This document summarizes all experimental phases executed for the Stage 3 benchmark. Every numerical figure in this document and accompanying reports is parsed directly from saved output logs and JSON artifact records.

### Execution Environment & Hardware Specifications
- **Compute Device:** NVIDIA GeForce RTX 3050 6GB Laptop GPU
- **Python Runtime:** Python 3.13.1 (64-bit)
- **Deep Learning Framework:** PyTorch 2.6.0+cu124 / CUDA 12.4
- **Random Seed Ensemble:** 3 independent seeds (`0, 1, 2`) reporting mean ± std

### Phase 0: Test Sanity Suite & Dataset Inventory
- **Pytest Verification:** 6/6 unit tests passing (`tests/test_sanity.py`) verifying POD energy conservation, Q-DEIM indexing uniqueness, Gappy reconstruction dimension matching, SensorMLP and DeepONet forward dimensions, and physics loss continuity/vorticity penalty correctness.
- **Dataset Inventory:** 136 full Navier-Stokes DNS cases cataloged (`results/case_inventory.csv`) spanning $Re \in [4, 1000]$, confirming the physical laminar Karman vortex shedding regime ($Re \in [100, 400]$) utilized across experiments.

### Phase 1: Sensor Budget Scaling Sweep & Gappy POD Rank Investigation
- **Mean-Flow Baseline ($p = 0$):** Rel $L_2 = 74.55\%$, Vorticity $\epsilon_\omega = 48.80\%$, Continuity $\mathcal{R}_{cont} = 0.1763$, Latency $< 0.1$ ms.
- **Nominal Gappy POD (r=5, Fixed $\lambda = 10^{-3}$):** Rel $L_2 = 17.36\%$, Vorticity $\epsilon_\omega = 21.23\%$.
- **Nominal Gappy POD (r=5, Tuned $\lambda = 1.0e-05$):** Rel $L_2 = 15.38\%$, Vorticity $\epsilon_\omega = 20.94\%$, Latency $= 0.18$ ms (CPU).
- **SensorMLP (Proposed SciML):** Rel $L_2 = 3.12 \pm 0.01\%$, Vorticity $\epsilon_\omega = 3.05 \pm 0.05\%$, Continuity $\mathcal{R}_{cont} = 0.1557$, Latency $= 0.72$ ms (GPU).
- **SensorMLP Invariance Mechanism:** SensorMLP maintains $\sim 3.12\%$ error across all $p \in [4, 128]$. In a limit-cycle laminar Karman wake ($Re=200$), the flow trajectory resides on a 1D periodic orbit in phase space; even 4 shear-layer sensors provide sufficient phase disambiguation for the 25.9M-parameter MLP to output the full matching flow field.
- **Gappy POD Rank Sweep ($p=16$):** As modal rank increases from $r=6$ to $r=20$, Gappy POD error drops from 14.43% to **8.72%** (truncation floor drops from 12.65% to 3.80%). At $r=40$, condition number degradation ($\kappa = 271.5$) increases error to 10.14%.
- **Gappy POD Rank Sweep ($p=32$):** With 96 measurement channels, condition number remains well-controlled ($\kappa = 20.7$). At $r=40$, Gappy POD achieves **2.86% error**, outperforming SensorMLP (3.12%).

### Phase 2: Sensor Placement Topology Benchmark
- At $p=8$: Uniform (21.12%) vs Random (20.01 ± 2.92%) vs **Q-DEIM (15.81%)** for Gappy POD.
- At $p=16$: Uniform (16.07%) vs Random (17.03 ± 1.53%) vs **Q-DEIM (15.38%)** for Gappy POD.
- At $p=32$: Uniform (15.35%) vs Random (16.02 ± 0.71%) vs **Q-DEIM (14.60%)** for Gappy POD.
- **Physical Takeaway:** Q-DEIM concentrates sensors along the boundary-layer separation points and downstream vortex street, maximizing interpolation matrix conditioning $\kappa(\mathbf{C}\mathbf{\Phi}_r)$ and outperforming spatial uniform gridding.

### Phase 3: Measurement Noise Robustness Sweep
- **Tikhonov Regularization Power:** At extreme noise (SNR = 10 dB), Gappy POD increases by only 0.73 percentage points (15.38% to 16.11%), demonstrating that optimal Tikhonov regularization ($\lambda = 10^{-4}$) effectively suppresses high-frequency sensor noise.
- **SensorMLP Noise Sensitivity:** Clean-trained SensorMLP degrades by nearly 2× from 3.12% to 6.00 ± 0.10%.
- **Noise-Augmented Training:** Reduces error at 10 dB to **4.25 ± 0.03%**, though it incurs a minor penalty under perfectly clean conditions (3.35% vs 3.12%).

### Phase 4: Reynolds Number Multi-Regime Generalization
- **Training Pool:** Combined snapshots across $Re \in \{100, 200, 300\}$.
- **In-Distribution ($Re = 100$):** Pooled Gappy POD = **0.60%** vs Pooled SensorMLP = 2.54% (Gappy POD is **4.23× better**).
- **In-Distribution ($Re = 200$):** Pooled Gappy POD = **2.37%** vs Pooled SensorMLP = 2.78% (Gappy POD is **1.17× better**).
- **Unseen Interpolation ($Re = 240$):** Pooled Gappy POD = **2.23%** vs Pooled SensorMLP = 3.44% (Gappy POD is **1.54× better**).
- **Unseen Extrapolation ($Re = 400$):** Pooled SensorMLP = **6.21%** vs Pooled Gappy POD = 8.13% (SensorMLP is **1.31× better**).
- **Physical Takeaway:** Linear modal interpolation across a multi-Reynolds library is remarkably accurate inside the convex hull ($Re \in [100, 300]$), outperforming neural networks. The non-linear inductive bias of SensorMLP proves advantageous strictly during out-of-distribution extrapolation ($Re=400$) where modal dynamics depart from the training subspace.

### Phase 5: Deep Learning & Operator Network Architectural Benchmark
- **Gappy POD (Linear ROM):** Rel $L_2 = 15.38\%$, Vorticity = 20.94%, Latency = 0.18 ms, Parameters = 0 (Analytic).
- **PODMLP (Latent SciML):** Rel $L_2 = 13.67 \pm 0.00\%$, Vorticity = 19.34%, Latency = 0.30 ms, Parameters = 23,941.
- **SensorMLP (Direct SciML):** Rel $L_2 = 3.12 \pm 0.01\%$, Vorticity = 3.05%, Latency = 0.72 ms, Parameters = 25,890,432.
- **DeepONet (Continuous Operator):** Rel $L_2 = 36.35 \pm 1.51\%$, Latency = 0.60 ms, Parameters = 106,563.
- **DeepONet Analysis:** DeepONet exhibits higher error (36.35%) due to parameter disparity (106k vs 25.9M in SensorMLP) and the representational bottleneck of rank-$q$ bilinear separation of variables $\sum b_k(y) t_k(x)$ on sharp localized shear layers without Fourier positional encodings. Evaluating on a 50% subsampled grid yields the exact same relative error (36.35%) because discrete relative $L_2$ error approximates the continuous functional $L_2$ norm, rather than proving spatial generalization.

### Phase 6: AirfRANS Incompressible Airfoils Benchmark
- **Data Source:** Genuine OpenFOAM incompressible RANS CFD simulations from the NeurIPS 2022 AirfRANS dataset spanning various NACA 4- and 5-digit airfoils at $Re \sim 10^6$ and $\alpha \in [-4.9^\circ, 14.9^\circ]$.
- **Extrapolation Challenge:** Lowest 75% AoA used for training (24 fit, 5 validation); highest 25% AoA ($\alpha \in [9.2^\circ, 14.9^\circ]$) reserved for testing, representing pure near-stall extrapolation.
- **Optimal Sensor Placement:** Q-DEIM placed **13/16 sensors directly in the leading-edge suction peak** ($x/c \in [0, 0.35]$), mirroring physical wind tunnel surface pressure taps.
- **Mean-Flow Baseline ($p=0$):** Rel $L_2 = 62.91\%$.
- **Gappy POD (Linear ROM):** Rel $L_2 = **45.19\%** (Sole model outperforming baseline, beating it by 17.7 percentage points).
- **PODMLP (Latent SciML):** Rel $L_2 = 62.90\%$ (Matches mean baseline).
- **DeepONet (Continuous Operator):** Rel $L_2 = 68.56\%$ (Fails to beat baseline).
- **SensorMLP (Direct SciML):** Rel $L_2 = 81.79\%$ (Fails to beat baseline; direct regression over-smoothes the suction peak without dense training coverage across separation angles).

### Master Quantitative Summary Table
Refer to `output/stage3/tables.md` and `output/stage3/tables.tex` for publication-ready LaTeX tables corresponding to Tables 1 through 7.