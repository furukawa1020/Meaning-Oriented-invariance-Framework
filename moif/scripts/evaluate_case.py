"""
Evaluate Invariance Breaking on CASE Dataset.
Compares 'bluVid' (Baseline) vs 'scary' (Stress equivalent) 
across all 30 subjects at 100Hz resolution.
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Add project root to sys.path
sys.path.append('.')

from moif.loaders.case import load_case
from moif.invariance.stats import permutation_test
from sklearn.neighbors import NearestNeighbors

print("Loading CASE dataset (interpolated 1000Hz synchronized to 100Hz)...")
# Note: Using the moved directory
df = load_case('data/case')

print(f"Total CASE samples loaded: {len(df)}")
subjects = df['subject_id'].unique()

# We use Valence/Arousal to show divergence despite physiological overlap
# Standardizing features (Z-score from baseline per subject)
features = ['ECG', 'GSR', 'BVP'] # Basic physiological features available in CASE
z_features = [f'{f}_Z' for f in features]

results = []

for sub_id in subjects:
    sub_df = df[df['subject_id'] == sub_id].copy()
    
    # CASE: 10: baseline, 1 & 2: stress (scary)
    b = sub_df[sub_df['label'] == 'baseline'].copy()
    s = sub_df[sub_df['label'] == 'stress'].copy()
    
    if b.empty or s.empty:
        print(f"Skipping {sub_id}: missing baseline or stress classes.")
        continue
        
    # Scale per subject baseline
    for col in features:
        m = b[col].mean()
        std = b[col].std()
        if std == 0: continue
        b.loc[:, f'{col}_Z'] = (b[col] - m) / std
        s.loc[:, f'{col}_Z'] = (s[col] - m) / std
        
    if f'{features[0]}_Z' not in b.columns: continue
    
    # 1. Overlap (Omega) calculation at r=1.0
    nn = NearestNeighbors(radius=1.0)
    b_vals = b[z_features].values
    if len(b_vals) > 10000:
        b_vals = b_vals[np.random.choice(len(b_vals), 10000, replace=False)]
    
    nn.fit(b_vals)
    
    s_vals = s[z_features].values
    if len(s_vals) > 10000:
        s_vals = s_vals[np.random.choice(len(s_vals), 10000, replace=False)]
        
    ind = nn.radius_neighbors(s_vals, return_distance=False)
    overlap_pct = np.mean([len(n) > 0 for n in ind]) * 100
    
    # 2. Subjective Divergence
    # In CASE, we have continuous valence/arousal
    mean_val_b = b['valence'].mean()
    mean_val_s = s['valence'].mean()
    val_diff = abs(mean_val_s - mean_val_b)
    
    results.append({
        'Subject': sub_id,
        'Overlap (%)': overlap_pct,
        'Valence Diff': val_diff
    })
    
    print(f"{sub_id}: Overlap={overlap_pct:.2f}%, Valence Diff={val_diff:.2f}")

res_df = pd.DataFrame(results).sort_values('Overlap (%)', ascending=False)
print("\n=== CASE Universal Distribution Overlap (Physiology: ECG, GSR, BVP) ===")
print(res_df.to_string())

res_df.to_csv("case_overlap_results.csv", index=False)
print("\nAnalysis complete. Results saved to case_overlap_results.csv.")
