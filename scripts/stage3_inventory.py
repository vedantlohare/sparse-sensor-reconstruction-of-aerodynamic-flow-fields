import os
import json
import csv
import numpy as np

def inventory_cases():
    raw_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'raw', 'cylinder')
    results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
    os.makedirs(results_dir, exist_ok=True)
    out_csv = os.path.join(results_dir, 'case_inventory.csv')
    
    cases = sorted([d for d in os.listdir(raw_dir) if os.path.isdir(os.path.join(raw_dir, d)) and d.startswith('case')])
    print(f"Found {len(cases)} cases in {raw_dir}")
    
    rows = []
    radii = set()
    x_bounds = set()
    y_bounds = set()
    
    for case_id in cases:
        case_path = os.path.join(raw_dir, case_id)
        json_path = os.path.join(case_path, 'case.json')
        u_path = os.path.join(case_path, 'u.npy')
        v_path = os.path.join(case_path, 'v.npy')
        
        with open(json_path, 'r') as f:
            meta = json.load(f)
            
        vel_in = float(meta['vel_in'])
        density = float(meta['density'])
        viscosity = float(meta['viscosity'])
        radius = float(meta['radius'])
        diameter = 2.0 * radius
        Re = (density * vel_in * diameter) / viscosity
        
        x_min, x_max = float(meta.get('x_min', -0.02)), float(meta.get('x_max', 0.16))
        y_min, y_max = float(meta.get('y_min', -0.06)), float(meta.get('y_max', 0.06))
        
        radii.add(radius)
        x_bounds.add((x_min, x_max))
        y_bounds.add((y_min, y_max))
        
        # Check array shape without loading entire array into memory if possible
        u_shape = np.load(u_path, mmap_mode='r').shape
        v_shape = np.load(v_path, mmap_mode='r').shape
        
        rows.append({
            'case_id': case_id,
            'Re': round(Re, 2),
            'radius': radius,
            'vel_in': vel_in,
            'density': density,
            'viscosity': viscosity,
            'x_min': x_min,
            'x_max': x_max,
            'y_min': y_min,
            'y_max': y_max,
            'u_shape': str(u_shape),
            'v_shape': str(v_shape)
        })
        
    with open(out_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
        
    print(f"Saved inventory to {out_csv}")
    
    re_values = [r['Re'] for r in rows]
    print(f"Re Range: min={min(re_values):.1f}, max={max(re_values):.1f}, unique Re count={len(set(re_values))}")
    print(f"Radius values found: {radii}")
    print(f"X bounds found: {x_bounds}")
    print(f"Y bounds found: {y_bounds}")
    print(f"Unique Re list (sorted): {sorted(list(set(re_values)))}")

if __name__ == '__main__':
    inventory_cases()
