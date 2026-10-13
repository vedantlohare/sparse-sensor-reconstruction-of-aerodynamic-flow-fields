# Stage 3 Experimental Benchmark Tables

## Table 1: Nominal Benchmark Reconstruction Performance ($p = 16$ Sensors, $Re \approx 200$)

| Model / Architecture | Sensors ($p$) | Rel. $L_2$ Error (%) | Vorticity Error $\epsilon_\omega$ (%) | Continuity Residual $\mathcal{R}_{cont}$ | Wall Pressure MAE | Latency (ms) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| Mean-Flow-Only Baseline | 0 | 74.55% | 48.80% | 0.1763 | 0.0552 | < 0.1 ms |
| Linear Grid Interpolation | 16 | 131.54% | 114.22% | 0.0567 | 0.8506 | ~ 15 ms |
| Gappy POD (Fixed $\lambda = 10^{-3}$) | 16 | 17.36% | 21.23% | 0.1695 | 0.0189 | 0.18 ms (CPU) |
| Gappy POD (Tuned $\lambda = 1.0e-05$) | 16 | 15.38% | 20.94% | 0.1716 | 0.0170 | 0.18 ms (CPU) |
| **SensorMLP (Proposed SciML)** | 16 | **3.12 ± 0.01%** | **3.05 ± 0.05%** | **0.1557 ± 0.0001** | **0.0026 ± 0.0001** | **0.72 ms (GPU)** |

## Table 2: Sensor Budget Scaling Law Sweep Across $p \in \{4, 8, 16, 32, 64, 128\}$

| Sensors ($p$) | Meas. Dims ($2p$) | Spatial Coverage | Interp Rel $L_2$ (%) | Gappy Fixed Rel $L_2$ (%) | Gappy Tuned Rel $L_2$ (%) | SensorMLP Rel $L_2$ (%) | Error Ratio (Gappy Tuned / MLP) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 4 | 12 | 0.049% | 141.08% | 31.08% | 20.67% | **3.12 ± 0.01%** | **6.63×** |
| 8 | 24 | 0.098% | 135.26% | 20.14% | 15.81% | **3.12 ± 0.02%** | **5.08×** |
| 16 | 48 | 0.195% | 131.54% | 17.36% | 15.38% | **3.12 ± 0.01%** | **4.93×** |
| 32 | 96 | 0.391% | 130.77% | 15.11% | 14.60% | **3.13 ± 0.01%** | **4.67×** |
| 64 | 192 | 0.781% | 144.36% | 15.36% | 15.05% | **3.13 ± 0.01%** | **4.80×** |
| 128 | 384 | 1.562% | 146.42% | 14.65% | 14.65% | **3.16 ± 0.01%** | **4.64×** |

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

| SNR (dB) | Gappy POD Tuned $\lambda$ | Gappy POD Rel $L_2$ (%) | Clean-Trained SensorMLP Rel $L_2$ (%) | Noise-Augmented SensorMLP Rel $L_2$ (%) |
|:---:|:---:|:---:|:---:|:---:|
| $\infty$ (Clean) | 1.0e-05 | 15.38% | 3.12 ± 0.01% | **3.35 ± 0.02%** |
| 40dB | 1.0e-05 | 15.38% | 3.12 ± 0.01% | **3.35 ± 0.02%** |
| 30dB | 1.0e-05 | 15.39% | 3.16 ± 0.01% | **3.36 ± 0.02%** |
| 20dB | 1.0e-04 | 15.48% | 3.53 ± 0.02% | **3.46 ± 0.02%** |
| 10dB | 1.0e-04 | 16.11% | 6.00 ± 0.10% | **4.25 ± 0.03%** |

## Table 5: Reynolds Number Cross-Regime Generalization ($p = 16$ Sensors)

| Target Regime | Reynolds ($Re$) | Gappy POD (Pooled) Rel $L_2$ (%) | SensorMLP (Pooled) Rel $L_2$ (%) | SensorMLP (Re=200 Only) Rel $L_2$ (%) | Generalization Advantage |
|:---|:---:|:---:|:---:|:---:|:---:|
| Train | 100 | 0.60% | **2.54 ± 0.00%** | 29.17% | **11.46×** |
| Train | 200 | 2.37% | **2.78 ± 0.00%** | 2.96% | **1.07×** |
| Interpolation | 240 | 2.23% | **3.44 ± 0.06%** | 7.34% | **2.14×** |
| Interpolation | 260 | 2.55% | **3.94 ± 0.10%** | 9.77% | **2.48×** |
| Train | 300 | 3.92% | **4.81 ± 0.14%** | 13.87% | **2.88×** |
| Extrapolation | 400 | 8.13% | **6.21 ± 0.00%** | 20.87% | **3.36×** |

## Table 6: Comprehensive Model Comparison ($p = 16$ Q-DEIM Sensors)

| Model Architecture | Rel $L_2$ Error (%) | Vorticity Error $\epsilon_\omega$ (%) | Continuity Residual $\mathcal{R}_{cont}$ | Wall Pressure MAE | Latency (ms) | Trainable Parameters |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| Gappy POD (Linear ROM) | 15.38% | 20.94% | 0.1716 | 0.0170 | 0.18 ms | 0 (Analytic) |
| PODMLP (Latent SciML) | 13.67 ± 0.00% | 19.34 ± 0.01% | 0.1653 | 0.0171 | 0.30 ms | 23,941 |
| **SensorMLP (Direct SciML)** | **3.12 ± 0.01%** | **3.05 ± 0.05%** | **0.1557** | **0.0026** | **0.72 ms** | **25,890,432** |
| DeepONet (Operator Net) | 36.35 ± 1.51% | 59.46 ± 2.66% | 0.1377 | 0.1355 | 0.60 ms | 106,563 |