# Stage 3 Experimental Benchmark Tables

## Table 1: Nominal Benchmark Reconstruction Performance ($p = 16$ Sensors, $Re \approx 200$)

| Model / Architecture | Sensors ($p$) | Rel. $L_2$ Error (%) | Vorticity Error $\epsilon_\omega$ (%) | Continuity Residual $\mathcal{R}_{cont}$ | Wall Pressure MAE | Latency (ms) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| Mean-Flow-Only Baseline | 0 | 74.55% | 48.80% | 0.1763 | 0.0552 | < 0.1 ms |
| Linear Grid Interpolation (Raw) | 16 | 131.54% | 114.22% | 0.0567 | 0.8506 | ~ 15 ms |
| Gappy POD (r=5, Fixed $\lambda = 10^{-3}$) | 16 | 17.36% | 21.23% | 0.1695 | 0.0189 | 0.18 ms (CPU) |
| Gappy POD (r=5, Tuned $\lambda = 1.0e-05$) | 16 | 15.38% | 20.94% | 0.1716 | 0.0170 | 0.18 ms (CPU) |
| **SensorMLP (Proposed SciML)** | 16 | **3.12 ± 0.01%** | **3.05 ± 0.05%** | **0.1557 ± 0.0001** | **0.0026 ± 0.0001** | **0.72 ms (GPU)** |

## Table 2: Sensor Budget Scaling Law Sweep Across $p \in \{4, 8, 16, 32, 64, 128\}$

| Sensors ($p$) | Meas. Dims ($3p$) | Spatial Coverage | Gappy POD (r=5) Fixed (%) | Gappy POD (r=5) Tuned (%) | SensorMLP Rel $L_2$ (%) | Error Ratio (Gappy Tuned / MLP) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 4 | 12 | 0.049% | 31.08% | 20.67% | **3.12 ± 0.01%** | **6.63×** |
| 8 | 24 | 0.098% | 20.14% | 15.81% | **3.12 ± 0.02%** | **5.08×** |
| 16 | 48 | 0.195% | 17.36% | 15.38% | **3.12 ± 0.01%** | **4.93×** |
| 32 | 96 | 0.391% | 15.11% | 14.60% | **3.13 ± 0.01%** | **4.67×** |
| 64 | 192 | 0.781% | 15.36% | 15.05% | **3.13 ± 0.01%** | **4.80×** |
| 128 | 384 | 1.562% | 14.65% | 14.65% | **3.16 ± 0.01%** | **4.64×** |

## Table 2B: Gappy POD Modal Rank Sweep: Basis Truncation vs. Conditioning ($Re \approx 200$)

| Sensors ($p$) | Modal Rank ($r$) | Energy (%) | Truncation Floor (%) | Gappy POD Rel $L_2$ (%) | Condition $\kappa(\mathbf{\Theta})$ | Tuned $\lambda$ | SensorMLP Rel $L_2$ (%) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 16 | 6 | 96.72% | 12.65% | 14.43% | 2.1 | 0 | 3.12 ± 0.01% |
| 16 | 10 | 98.61% | 8.01% | 10.94% | 2.6 | 0 | 3.12 ± 0.01% |
| 16 | 20 | 99.71% | 3.80% | 8.72% | 11.0 | 0 | 3.12 ± 0.01% |
| 16 | 40 | 99.97% | 1.45% | 10.14% | 271.5 | 1e-06 | 3.12 ± 0.01% |
| 32 | 6 | 96.72% | 12.65% | 13.77% | 1.9 | 0 | 3.13 ± 0.01% |
| 32 | 10 | 98.61% | 8.01% | 9.78% | 2.4 | 0 | 3.13 ± 0.01% |
| 32 | 20 | 99.71% | 3.80% | 5.23% | 4.3 | 0 | 3.13 ± 0.01% |
| 32 | 40 | 99.97% | 1.45% | **2.86%** | 20.7 | 1e-05 | 3.13 ± 0.01% |

## Table 3: Sensor Placement Strategy Comparison ($p \in \{8, 16, 32\}$)

| Sensors ($p$) | Placement Strategy | Gappy POD Rel $L_2$ (%) | SensorMLP Rel $L_2$ (%) |
|:---:|:---|:---:|:---:|
| 8 | Uniform Grid | 21.12% | 3.21 ± 0.00% |
| 8 | Random Placement | 20.01 ± 2.92% | 3.20 ± 0.04% |
| 8 | **Q-DEIM (Tailored)** | **15.81%** | **3.12 ± 0.02%** |
| 16 | Uniform Grid | 16.07% | 3.16 ± 0.00% |
| 16 | Random Placement | 17.03 ± 1.53% | 3.16 ± 0.03% |
| 16 | **Q-DEIM (Tailored)** | **15.38%** | **3.12 ± 0.01%** |
| 32 | Uniform Grid | 15.35% | 3.14 ± 0.01% |
| 32 | Random Placement | 16.02 ± 0.71% | 3.16 ± 0.01% |
| 32 | **Q-DEIM (Tailored)** | **14.60%** | **3.13 ± 0.01%** |

## Table 4: Measurement Noise Robustness ($p = 16$ Q-DEIM Sensors)

| SNR (dB) | Gappy POD Tuned $\lambda$ | Gappy POD Rel $L_2$ (%) | Clean-Trained SensorMLP Rel $L_2$ (%) | Noise-Augmented SensorMLP Rel $L_2$ (%) | Observation / Dynamics |
|:---:|:---:|:---:|:---:|:---:|:---|
| $\infty$ (Clean) | 1.0e-05 | 15.38% | 3.12 ± 0.01% | **3.35 ± 0.02%** | Baseline |
| 40dB | 1.0e-05 | 15.38% | 3.12 ± 0.01% | **3.35 ± 0.02%** | Stable |
| 30dB | 1.0e-05 | 15.39% | 3.16 ± 0.01% | **3.36 ± 0.02%** | Stable |
| 20dB | 1.0e-04 | 15.48% | 3.53 ± 0.02% | **3.46 ± 0.02%** | MLP sensitive |
| 10dB | 1.0e-04 | 16.11% | 6.00 ± 0.10% | **4.25 ± 0.03%** | Tikhonov dampens noise (+0.73%), MLP degrades 2x |

## Table 5: Reynolds Number Cross-Regime Generalization ($p = 16$ Sensors)

| Target Regime | Reynolds ($Re$) | Pooled Gappy POD Rel $L_2$ (%) | Pooled SensorMLP Rel $L_2$ (%) | Generalization Comparison | Single-$Re$ ($Re=200$) MLP (%) |
|:---|:---:|:---:|:---:|:---:|:---:|
| Train | 100 | **0.60%** | 2.54 ± 0.00% | Gappy POD (4.23× better) | 29.17% |
| Train | 200 | **2.37%** | 2.78 ± 0.00% | Gappy POD (1.17× better) | 2.96% |
| Interpolation | 240 | **2.23%** | 3.44 ± 0.06% | Gappy POD (1.54× better) | 7.34% |
| Interpolation | 260 | **2.55%** | 3.94 ± 0.10% | Gappy POD (1.54× better) | 9.77% |
| Train | 300 | **3.92%** | 4.81 ± 0.14% | Gappy POD (1.23× better) | 13.87% |
| Extrapolation | 400 | 8.13% | **6.21 ± 0.00%** | **SensorMLP (1.31× better)** | 20.87% |

## Table 6: Comprehensive Model Comparison ($p = 16$ Q-DEIM Sensors)

| Model Architecture | Rel $L_2$ Error (%) | Vorticity Error $\epsilon_\omega$ (%) | Continuity Residual $\mathcal{R}_{cont}$ | Wall Pressure MAE | Latency (ms) | Trainable Parameters |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| Gappy POD (Linear ROM) | 15.38% | 20.94% | 0.1716 | 0.0170 | 0.18 ms | 0 (Analytic) |
| PODMLP (Latent SciML) | 13.67 ± 0.00% | 19.34 ± 0.01% | 0.1653 | 0.0171 | 0.30 ms | 23,941 |
| **SensorMLP (Direct SciML)** | **3.12 ± 0.01%** | **3.05 ± 0.05%** | **0.1557** | **0.0026** | **0.72 ms** | **25,890,432** |
| DeepONet (Operator Net) | 36.35 ± 1.51% | 59.46 ± 2.66% | 0.1377 | 0.1355 | 0.60 ms | 106,563 |

## Table 7: AirfRANS Incompressible Airfoils Benchmark ($p = 16$ Q-DEIM Sensors)

| Model Architecture | Rel $L_2$ Error (%) | Vorticity Error $\epsilon_\omega$ (%) | Continuity Residual $\mathcal{R}_{cont}$ | Latency (ms) | Generalization Status |
|:---|:---:|:---:|:---:|:---:|:---|
| Mean-Flow Baseline ($p=0$) | 62.91% | — | — | < 0.1 ms | Reference Baseline |
| **Gappy POD (Linear ROM)** | **45.19%** | 140.18% | 289.12 | 0.10 ms | **Sole model beating baseline (-17.7%)** |
| PODMLP (Latent SciML) | 62.90% | 99.86% | 52.43 | ~0.40 ms | Matches baseline |
| DeepONet (Operator Net) | 68.56% | 219.93% | 85.16 | ~1.20 ms | Fails baseline (+5.7% error) |
| SensorMLP (Direct SciML) | 81.79% | 99.42% | 56.00 | 0.73 ms | Fails baseline (+18.9% error) |