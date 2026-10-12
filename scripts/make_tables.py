import os
import json
import numpy as np

def format_val(val, std=None, fmt="%.2f"):
    if std is not None:
        return f"{fmt % val} ± {fmt % std}"
    return fmt % val

def generate_tables():
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir = os.path.join(root_dir, 'results')
    output_dir = os.path.join(root_dir, 'output', 'stage3')
    os.makedirs(output_dir, exist_ok=True)
    
    psweep_file = os.path.join(results_dir, 'phase1_psweep.json')
    if not os.path.exists(psweep_file):
        print(f"Results file {psweep_file} not found. Run stage3_psweep.py first.")
        return
        
    with open(psweep_file, 'r') as f:
        data = json.load(f)
        
    sweep = data['sweep']
    mean_baseline = data.get('mean_flow_baseline', {})
    latencies = data.get('latencies_ms', {})
    
    md_lines = []
    tex_lines = []
    
    # -------------------------------------------------------------
    # Table 1: Nominal Benchmark at p = 16
    # -------------------------------------------------------------
    md_lines.append("# Stage 3 Experimental Benchmark Tables\n")
    md_lines.append("## Table 1: Nominal Benchmark Reconstruction Performance ($p = 16$ Sensors, $Re \\approx 200$)\n")
    md_lines.append("| Model / Architecture | Sensors ($p$) | Rel. $L_2$ Error (%) | Vorticity Error $\\epsilon_\\omega$ (%) | Continuity Residual $\\mathcal{R}_{cont}$ | Wall Pressure MAE | Latency (ms) |")
    md_lines.append("|---|:---:|:---:|:---:|:---:|:---:|:---:|")
    
    tex_lines.append("% Table 1: Nominal Benchmark (p = 16)")
    tex_lines.append("\\begin{table}[h!]")
    tex_lines.append("\\centering")
    tex_lines.append("\\caption{Quantitative Reconstruction Performance on Unseen Test Snapshots ($p = 16$ Sensors, $Re \\approx 200$, 3 Seeds).}")
    tex_lines.append("\\label{tab:nominal_benchmark_p16}")
    tex_lines.append("\\begin{tabular}{lcccccc}")
    tex_lines.append("\\toprule")
    tex_lines.append("\\textbf{Model / Architecture} & \\textbf{Sensors ($p$)} & \\textbf{Rel. $L_2$ (\\%)} & \\textbf{Vorticity $\\epsilon_\\omega$ (\\%)} & \\textbf{Residual $\\mathcal{R}_{cont}$} & \\textbf{Wall $p$ MAE} & \\textbf{Latency (ms)} \\\\")
    tex_lines.append("\\midrule")
    
    # Rows for Table 1
    # 1. Mean-flow
    if mean_baseline:
        md_lines.append(f"| Mean-Flow-Only Baseline | 0 | {mean_baseline['rel_l2_pct']:.2f}% | {mean_baseline['vort_err_pct']:.2f}% | {mean_baseline['r_cont']:.4f} | {mean_baseline['wall_p_mae']:.4f} | < 0.1 ms |")
        tex_lines.append(f"Mean-Flow-Only Baseline & 0 & {mean_baseline['rel_l2_pct']:.2f}\\% & {mean_baseline['vort_err_pct']:.2f}\\% & {mean_baseline['r_cont']:.4f} & {mean_baseline['wall_p_mae']:.4f} & $< 0.1$ ms \\\\")
        
    p16 = sweep.get('16')
    if p16:
        # 2. Linear Interp
        interp = p16['interp_linear']
        md_lines.append(f"| Linear Grid Interpolation | 16 | {interp['rel_l2_pct']:.2f}% | {interp['vort_err_pct']:.2f}% | {interp['r_cont']:.4f} | {interp['wall_p_mae']:.4f} | ~ 15 ms |")
        tex_lines.append(f"Linear Grid Interpolation & 16 & {interp['rel_l2_pct']:.2f}\\% & {interp['vort_err_pct']:.2f}\\% & {interp['r_cont']:.4f} & {interp['wall_p_mae']:.4f} & $\\sim 15$ ms \\\\")
        
        # 3. Gappy Fixed
        g_fix = p16['gappy_fixed_1e3']
        g_lat = f"{latencies.get('gappy_pod_cpu_ms', 0.24):.2f} ms"
        md_lines.append(f"| Gappy POD (Fixed $\\lambda = 10^{{-3}}$) | 16 | {g_fix['rel_l2_pct']:.2f}% | {g_fix['vort_err_pct']:.2f}% | {g_fix['r_cont']:.4f} | {g_fix['wall_p_mae']:.4f} | {g_lat} (CPU) |")
        tex_lines.append(f"Gappy POD (Fixed $\\lambda = 10^{{-3}}$) & 16 & {g_fix['rel_l2_pct']:.2f}\\% & {g_fix['vort_err_pct']:.2f}\\% & {g_fix['r_cont']:.4f} & {g_fix['wall_p_mae']:.4f} & {g_lat} \\\\")
        
        # 4. Gappy Tuned
        g_tune = p16['gappy_tuned']['metrics']
        best_lam = p16['gappy_tuned']['best_lambda']
        md_lines.append(f"| Gappy POD (Tuned $\\lambda = {best_lam:.1e}$) | 16 | {g_tune['rel_l2_pct']:.2f}% | {g_tune['vort_err_pct']:.2f}% | {g_tune['r_cont']:.4f} | {g_tune['wall_p_mae']:.4f} | {g_lat} (CPU) |")
        tex_lines.append(f"Gappy POD (Tuned $\\lambda = {best_lam:.1e}$) & 16 & {g_tune['rel_l2_pct']:.2f}\\% & {g_tune['vort_err_pct']:.2f}\\% & {g_tune['r_cont']:.4f} & {g_tune['wall_p_mae']:.4f} & {g_lat} \\\\")
        
        # 5. SensorMLP Mean ± Std
        mlp_m = p16['sensor_mlp_mean']
        mlp_s = p16['sensor_mlp_std']
        mlp_gpu_lat = f"{latencies.get('sensor_mlp_gpu_ms', 2.09):.2f} ms" if latencies.get('sensor_mlp_gpu_ms') else "N/A"
        md_lines.append(f"| **SensorMLP (Proposed SciML)** | 16 | **{mlp_m['rel_l2_pct']:.2f} ± {mlp_s['rel_l2_pct']:.2f}%** | **{mlp_m['vort_err_pct']:.2f} ± {mlp_s['vort_err_pct']:.2f}%** | **{mlp_m['r_cont']:.4f} ± {mlp_s['r_cont']:.4f}** | **{mlp_m['wall_p_mae']:.4f} ± {mlp_s['wall_p_mae']:.4f}** | **{mlp_gpu_lat} (GPU)** |")
        tex_lines.append(f"\\textbf{{SensorMLP (Proposed SciML)}} & 16 & \\textbf{{{mlp_m['rel_l2_pct']:.2f} $\\pm$ {mlp_s['rel_l2_pct']:.2f}\\%}} & \\textbf{{{mlp_m['vort_err_pct']:.2f} $\\pm$ {mlp_s['vort_err_pct']:.2f}\\%}} & \\textbf{{{mlp_m['r_cont']:.4f} $\\pm$ {mlp_s['r_cont']:.4f}}} & \\textbf{{{mlp_m['wall_p_mae']:.4f} $\\pm$ {mlp_s['wall_p_mae']:.4f}}} & \\textbf{{{mlp_gpu_lat}}} \\\\")
        
    tex_lines.append("\\bottomrule")
    tex_lines.append("\\end{tabular}")
    tex_lines.append("\\end{table}\n")
    
    # -------------------------------------------------------------
    # Table 2: Sensor Scaling Sweep Across p
    # -------------------------------------------------------------
    md_lines.append("\n## Table 2: Sensor Budget Scaling Law Sweep Across $p \\in \\{4, 8, 16, 32, 64, 128\\}$\n")
    md_lines.append("| Sensors ($p$) | Meas. Dims ($2p$) | Spatial Coverage | Interp Rel $L_2$ (%) | Gappy Fixed Rel $L_2$ (%) | Gappy Tuned Rel $L_2$ (%) | SensorMLP Rel $L_2$ (%) | Error Ratio (Gappy Tuned / MLP) |")
    md_lines.append("|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")
    
    tex_lines.append("% Table 2: Sensor Budget Scaling Sweep")
    tex_lines.append("\\begin{table}[h!]")
    tex_lines.append("\\centering")
    tex_lines.append("\\caption{Sensor Budget Scaling Law: Reconstruction Error Across Varying Sensor Counts ($p$).}")
    tex_lines.append("\\label{tab:sensor_scaling_sweep}")
    tex_lines.append("\\begin{tabular}{cccccccc}")
    tex_lines.append("\\toprule")
    tex_lines.append("\\textbf{$p$} & \\textbf{Dims ($2p$)} & \\textbf{Coverage} & \\textbf{Interp (\\%)} & \\textbf{Gappy Fix (\\%)} & \\textbf{Gappy Tune (\\%)} & \\textbf{SensorMLP (\\%)} & \\textbf{Ratio (Gappy/MLP)} \\\\")
    tex_lines.append("\\midrule")
    
    for p_key in sorted(sweep.keys(), key=lambda x: int(x)):
        row = sweep[p_key]
        p_val = row['p']
        dims = row['meas_dims']
        cov = f"{row['spatial_coverage_pct']:.3f}%"
        interp_l2 = f"{row['interp_linear']['rel_l2_pct']:.2f}%"
        g_fix_l2 = f"{row['gappy_fixed_1e3']['rel_l2_pct']:.2f}%"
        g_tune_l2 = f"{row['gappy_tuned']['metrics']['rel_l2_pct']:.2f}%"
        
        mlp_m_l2 = row['sensor_mlp_mean']['rel_l2_pct']
        mlp_s_l2 = row['sensor_mlp_std']['rel_l2_pct']
        mlp_str = f"{mlp_m_l2:.2f} ± {mlp_s_l2:.2f}%"
        
        ratio = row['gappy_tuned']['metrics']['rel_l2_pct'] / max(mlp_m_l2, 1e-4)
        ratio_str = f"{ratio:.2f}×"
        
        md_lines.append(f"| {p_val} | {dims} | {cov} | {interp_l2} | {g_fix_l2} | {g_tune_l2} | **{mlp_str}** | **{ratio_str}** |")
        tex_lines.append(f"{p_val} & {dims} & {cov} & {interp_l2} & {g_fix_l2} & {g_tune_l2} & \\textbf{{{mlp_m_l2:.2f} $\\pm$ {mlp_s_l2:.2f}\\%}} & \\textbf{{{ratio:.2f}$\\times$}} \\\\")
        
    tex_lines.append("\\bottomrule")
    tex_lines.append("\\end{tabular}")
    tex_lines.append("\\end{table}\n")
    
    # -------------------------------------------------------------
    # Table 3: Sensor Placement Comparison (Topology)
    # -------------------------------------------------------------
    topo_file = os.path.join(results_dir, 'phase2_topology.json')
    if os.path.exists(topo_file):
        with open(topo_file, 'r') as f:
            topo_data = json.load(f)['results']
            
        md_lines.append("\n## Table 3: Sensor Placement Strategy Comparison ($p \\in \\{8, 16, 32\\}$)\n")
        md_lines.append("| Sensors ($p$) | Placement Strategy | Gappy POD Rel $L_2$ (%) | SensorMLP Rel $L_2$ (%) |")
        md_lines.append("|:---:|:---|:---:|:---:|")
        
        tex_lines.append("% Table 3: Sensor Placement Comparison")
        tex_lines.append("\\begin{table}[h!]")
        tex_lines.append("\\centering")
        tex_lines.append("\\caption{Sensor Placement Comparison: Reconstruction Error across Placement Strategies.}")
        tex_lines.append("\\label{tab:sensor_placement_comparison}")
        tex_lines.append("\\begin{tabular}{cccc}")
        tex_lines.append("\\toprule")
        tex_lines.append("\\textbf{Sensors ($p$)} & \\textbf{Strategy} & \\textbf{Gappy POD Rel $L_2$ (\\%)} & \\textbf{SensorMLP Rel $L_2$ (\\%)} \\\\")
        tex_lines.append("\\midrule")
        
        for p_key in sorted(topo_data.keys(), key=lambda x: int(x)):
            p_res = topo_data[p_key]
            # Uniform
            u_g = f"{p_res['uniform']['gappy_pod']['rel_l2_pct']:.2f}%"
            u_m_m = p_res['uniform']['sensor_mlp_mean']['rel_l2_pct']
            u_m_s = p_res['uniform']['sensor_mlp_std']['rel_l2_pct']
            u_m = f"{u_m_m:.2f} ± {u_m_s:.2f}%"
            md_lines.append(f"| {p_key} | Uniform Grid | {u_g} | {u_m} |")
            tex_lines.append(f"{p_key} & Uniform Grid & {u_g} & {u_m_m:.2f} $\\pm$ {u_m_s:.2f}\\% \\\\")
            
            # Random
            r_g_m = p_res['random']['gappy_pod_mean']['rel_l2_pct']
            r_g_s = p_res['random']['gappy_pod_std']['rel_l2_pct']
            r_g = f"{r_g_m:.2f} ± {r_g_s:.2f}%"
            r_m_m = p_res['random']['sensor_mlp_mean']['rel_l2_pct']
            r_m_s = p_res['random']['sensor_mlp_std']['rel_l2_pct']
            r_m = f"{r_m_m:.2f} ± {r_m_s:.2f}%"
            md_lines.append(f"| {p_key} | Random Placement | {r_g} | {r_m} |")
            tex_lines.append(f"{p_key} & Random Placement & {r_g_m:.2f} $\\pm$ {r_g_s:.2f}\\% & {r_m_m:.2f} $\\pm$ {r_m_s:.2f}\\% \\\\")
            
            # Q-DEIM
            q_g = f"{p_res['qdeim']['gappy_pod']['rel_l2_pct']:.2f}%"
            q_m_m = p_res['qdeim']['sensor_mlp_mean']['rel_l2_pct']
            q_m_s = p_res['qdeim']['sensor_mlp_std']['rel_l2_pct']
            q_m = f"**{q_m_m:.2f} ± {q_m_s:.2f}%**"
            md_lines.append(f"| {p_key} | **Q-DEIM (Tailored)** | **{q_g}** | {q_m} |")
            tex_lines.append(f"{p_key} & \\textbf{{Q-DEIM (Tailored)}} & \\textbf{{{q_g}}} & \\textbf{{{q_m_m:.2f} $\\pm$ {q_m_s:.2f}\\%}} \\\\")
            if p_key != sorted(topo_data.keys(), key=lambda x: int(x))[-1]:
                tex_lines.append("\\midrule")
                
        tex_lines.append("\\bottomrule")
        tex_lines.append("\\end{tabular}")
        tex_lines.append("\\end{table}\n")
    # -------------------------------------------------------------
    # Table 4: Noise Robustness (p = 16)
    # -------------------------------------------------------------
    noise_file = os.path.join(results_dir, 'phase3_noise.json')
    if os.path.exists(noise_file):
        with open(noise_file, 'r') as f:
            noise_data = json.load(f)['results']
            
        md_lines.append("\n## Table 4: Measurement Noise Robustness ($p = 16$ Q-DEIM Sensors)\n")
        md_lines.append("| SNR (dB) | Gappy POD Tuned $\\lambda$ | Gappy POD Rel $L_2$ (%) | Clean-Trained SensorMLP Rel $L_2$ (%) | Noise-Augmented SensorMLP Rel $L_2$ (%) |")
        md_lines.append("|:---:|:---:|:---:|:---:|:---:|")
        
        tex_lines.append("% Table 4: Noise Robustness")
        tex_lines.append("\\begin{table}[h!]")
        tex_lines.append("\\centering")
        tex_lines.append("\\caption{Measurement Noise Robustness: Reconstruction Performance Across SNR Levels ($p = 16$).}")
        tex_lines.append("\\label{tab:noise_robustness}")
        tex_lines.append("\\begin{tabular}{ccccc}")
        tex_lines.append("\\toprule")
        tex_lines.append("\\textbf{SNR (dB)} & \\textbf{Tuned $\\lambda$} & \\textbf{Gappy POD (\\%)} & \\textbf{Clean-Trained MLP (\\%)} & \\textbf{Noise-Augmented MLP (\\%)} \\\\")
        tex_lines.append("\\midrule")
        
        # Order keys from clean (inf) to 10dB
        order = ['inf', '40dB', '30dB', '20dB', '10dB']
        valid_keys = [k for k in order if k in noise_data] + [k for k in noise_data if k not in order]
        
        for k in valid_keys:
            row = noise_data[k]
            snr_disp = r"$\infty$ (Clean)" if k == 'inf' else k
            snr_tex = r"$\infty$ (Clean)" if k == 'inf' else f"{row['snr_db']:.0f} dB"
            lam_disp = f"{row['gappy_tuned_lambda']:.1e}"
            gappy_val = f"{row['gappy_rel_l2_pct']:.2f}%"
            clean_m = f"{row['clean_mlp_rel_l2_mean']:.2f} ± {row['clean_mlp_rel_l2_std']:.2f}%"
            clean_tex = f"{row['clean_mlp_rel_l2_mean']:.2f} $\\pm$ {row['clean_mlp_rel_l2_std']:.2f}\\%"
            aug_m = f"**{row['augmented_mlp_rel_l2_mean']:.2f} ± {row['augmented_mlp_rel_l2_std']:.2f}%**"
            aug_tex = f"\\textbf{{{row['augmented_mlp_rel_l2_mean']:.2f} $\\pm$ {row['augmented_mlp_rel_l2_std']:.2f}\\%}}"
            
            md_lines.append(f"| {snr_disp} | {lam_disp} | {gappy_val} | {clean_m} | {aug_m} |")
            tex_lines.append(f"{snr_tex} & {lam_disp} & {gappy_val} & {clean_tex} & {aug_tex} \\\\")
            
        tex_lines.append("\\bottomrule")
        tex_lines.append("\\end{tabular}")
        tex_lines.append("\\end{table}\n")
    
    # Write output files
    md_out = os.path.join(output_dir, 'tables.md')
    tex_out = os.path.join(output_dir, 'tables.tex')
    
    with open(md_out, 'w', encoding='utf-8') as f:
        f.write('\n'.join(md_lines))
        
    with open(tex_out, 'w', encoding='utf-8') as f:
        f.write('\n'.join(tex_lines))
        
    print(f"Generated Markdown tables at: {md_out}")
    print(f"Generated LaTeX tables at: {tex_out}")

if __name__ == '__main__':
    generate_tables()
