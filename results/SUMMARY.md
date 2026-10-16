# Stage 3 Comprehensive Empirical Results Summary
## Team VRIGHT BROTHERS — AE646 Scientific Machine Learning for Fluid Mechanics

This document summarizes all experimental phases executed for the Stage 3 benchmark. Every numerical figure in this document and accompanying reports is parsed directly from saved output logs and JSON artifact records.

### Execution Environment & Hardware Specifications
- **Compute Device:** NVIDIA GeForce RTX 3050 6GB Laptop GPU
- **Python Runtime:** Python 3.13.1 (64-bit)
- **Deep Learning Framework:** PyTorch 2.6.0+cu124 / CUDA 12.4
- **Random Seed Ensemble:** 3 independent seeds (`0, 1, 2`) reporting mean ± std

### Phase 0: Test Sanity Suite & Dataset Inventory
- **Pytest Verification:** 5/5 unit tests passing (`tests/test_sanity.py`) verifying POD energy conservation, Q-DEIM indexing uniqueness, Gappy reconstruction dimension matching, SensorMLP forward dimensions, and physics loss continuity/vorticity penalty correctness.
- **Dataset Inventory:** 136 full Navier-Stokes DNS cases cataloged (`results/case_inventory.csv`) spanning $Re \in [4, 1000]$, confirming the physical laminar Karman vortex shedding regime ($Re \in [100, 400]$) utilized across experiments.

### Phase 1: Sensor Budget Scaling Sweep & Nominal Benchmark ($p = 16$)
- **Mean-Flow Baseline ($p = 0$):** Rel $L_2 = 74.55\%$, Vorticity $\epsilon_\omega = 48.80\%$, Continuity $\mathcal{R}_{cont} = 0.1763$, Latency $< 0.1$ ms.
- **Linear Interpolation ($p = 16$):** Rel $L_2 = 131.54\%$, Vorticity $\epsilon_\omega = 114.22\%$, Continuity $\mathcal{R}_{cont} = 0.0567$.
- **Gappy POD (Fixed $\lambda = 10^{-3}$):** Rel $L_2 = 17.36\%$, Vorticity $\epsilon_\omega = 21.23\%$.
- **Gappy POD (Tuned $\lambda = 1.0e-05$):** Rel $L_2 = 15.38\%$, Vorticity $\epsilon_\omega = 20.94\%$, Latency $= 0.18$ ms (CPU).
- **SensorMLP (Proposed SciML):** Rel $L_2 = 3.12 \pm 0.01\%$, Vorticity $\epsilon_\omega = 3.05 \pm 0.05\%$, Continuity $\mathcal{R}_{cont} = 0.1557$, Latency $= 0.72$ ms (GPU).
- **Budget Scaling Trend:** SensorMLP reconstructs the full wake with $\sim 3.12\%$ error even at extreme sparsity ($p = 4$, $0.049\%$ spatial coverage), outperforming Gappy POD by **4.93×** at $p=16$ and **6.63×** at $p=4$.

### Phase 2: Sensor Placement Topology Benchmark
- At $p=8$: Uniform (21.12%) vs Random (20.01 ± 2.92%) vs **Q-DEIM (15.81%)** for Gappy POD.
- At $p=16$: Uniform (16.07%) vs Random (17.03 ± 1.53%) vs **Q-DEIM (15.38%)** for Gappy POD.
- At $p=32$: Uniform (15.35%) vs Random (16.02 ± 0.71%) vs **Q-DEIM (14.60%)** for Gappy POD.
- **Physical Takeaway:** Q-DEIM concentrates sensors along the boundary-layer separation points and downstream vortex street, maximizing interpolation matrix conditioning $\kappa(\mathbf{C}\mathbf{\Phi}_r)$ and outperforming spatial uniform gridding.

### Phase 3: Measurement Noise Robustness Sweep
- Optimal Tikhonov regularization $\lambda$ increases monotonically from $10^{-5}$ at clean/30dB to $10^{-4}$ at 20dB/10dB.
- At high noise (SNR = 10 dB), clean-trained SensorMLP degrades to 6.00 ± 0.10%.
- Noise-augmented SensorMLP reduces error to **4.25 ± 0.03%**, achieving a **29.2% error reduction** under severe sensor noise.

### Phase 4: Reynolds Number Multi-Regime Generalization
- **Training Pool:** Combined snapshots across $Re \in \{100, 200, 300\}$.
- In-Distribution ($Re = 100$): Pooled MLP = **2.54%** vs Single-$Re$ = 29.17% (**11.5× better**).
- Unseen Interpolation ($Re = 240$): Pooled MLP = **3.44%** vs Single-$Re$ = 7.34% (**2.1× better**).
- Unseen Extrapolation ($Re = 400$): Pooled MLP = **6.21%** vs Gappy POD = 8.13% vs Single-$Re$ = 20.87% (**3.4× better**).

### Phase 5: Deep Learning & Operator Network Architectural Benchmark
- **Gappy POD (Linear ROM):** Rel $L_2 = 15.38\%$, Vorticity = 20.94%, Latency = 0.18 ms, Parameters = 0 (Analytic).
- **PODMLP (Latent SciML):** Rel $L_2 = 13.67 \pm 0.00\%$, Vorticity = 19.34%, Latency = 0.30 ms, Parameters = 23,941.
- **SensorMLP (Direct SciML):** Rel $L_2 = 3.12 \pm 0.01\%$, Vorticity = 3.05%, Latency = 0.72 ms, Parameters = 25,890,432.
- **DeepONet (Continuous Operator):** Rel $L_2 = 36.35 \pm 1.51\%$, 50% Mesh Subsample Error = 36.36%, Latency = 0.60 ms, Parameters = 106,563.

### Phase 6: Transonic/Subsonic NACA 0012 Airfoil Benchmark (AirfRANS CFD)
- **Data Source:** Genuine OpenFOAM RANS CFD solutions from the NeurIPS 2022 AirfRANS dataset ($Re \sim 10^6$, $\alpha \in [-5^\circ, 15^\circ]$).
- **Sensor Physics:** Q-DEIM placed **13/16 sensors along the leading-edge suction peak** ($x/c \in [0, 0.35]$), mirroring physical wind tunnel surface pressure taps.
- **Mean-Flow Baseline ($p=0$):** Rel $L_2 = 62.91\%$.
- **Gappy POD (Linear ROM):** Rel $L_2 = 45.19\%$, Vorticity $\epsilon_\omega = 140.18\%$, Continuity $\mathcal{R}_{cont} = 289.1168$.
- **PODMLP (Latent SciML):** Rel $L_2 = 62.90\%$, Vorticity $\epsilon_\omega = 99.86\%$, Continuity $\mathcal{R}_{cont} = 52.4314$.
- **DeepONet (Continuous Operator):** Rel $L_2 = 68.56\%$, Vorticity $\epsilon_\omega = 219.93\%$, Continuity $\mathcal{R}_{cont} = 85.1604$.
- **SensorMLP (Direct SciML):** Rel $L_2 = **81.79\%**, Vorticity $\epsilon_\omega = **99.42\%**, Continuity $\mathcal{R}_{cont} = **55.9995**.

### Master Quantitative Summary Table
Refer to `output/stage3/tables.md` and `output/stage3/tables.tex` for publication-ready LaTeX tables corresponding to Tables 1 through 7.