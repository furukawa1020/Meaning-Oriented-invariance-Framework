import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Add project root to sys.path
sys.path.append('.')

from moif.loaders.wesad import load_wesad
from moif.invariance.banding import apply_banding
from moif.invariance.stats import permutation_test, apply_fdr

print("Loading WESAD dataset with Advanced Sliding Windows (60s)... (this will take a few minutes)")
df = load_wesad('data/wesad/WESAD', window_size_sec=60, stride_sec=10)

all_classes = ['baseline', 'amusement', 'meditation', 'stress']

# Define banding based on advanced physiological features extracted via neurokit2
banding_cfgs = [
    {
        "feature": "HRV_RMSSD",
        "name": "Low HRV (Parasympathetic Withdrawal, Z < -0.5)",
        "config": {"mode": "norm", "norm": {"method": "z", "z_low": -10.0, "z_high": -0.5}}
    },
    {
        "feature": "HRV_LFHF",
        "name": "High LF/HF (Sympathetic Dominance, Z > 0.5)",
        "config": {"mode": "norm", "norm": {"method": "z", "z_low": 0.5, "z_high": 10.0}}
    },
    {
        "feature": "EDA_Tonic_Mean",
        "name": "High SCL (High General Arousal, Z > 0.5)",
        "config": {"mode": "norm", "norm": {"method": "z", "z_low": 0.5, "z_high": 10.0}}
    },
    {
        "feature": "EDA_Phasic_Mean",
        "name": "High SCR (High Phasic Reactivity, Z > 0.5)",
        "config": {"mode": "norm", "norm": {"method": "z", "z_low": 0.5, "z_high": 10.0}}
    }
]

subjects = df['subject_id'].unique()
results = []

for b_cfg in banding_cfgs:
    print(f"\nAnalyzing State: {b_cfg['name']}")
    feat_col = b_cfg['feature']
    
    df_banded = apply_banding(df, b_cfg['config'], feature_col=feat_col)
    df_in_band = df_banded[df_banded['in_band']]
    
    # Test Invariance Breaking across Subject IDs
    for i in range(len(subjects)):
        for j in range(i + 1, len(subjects)):
            s1 = subjects[i]
            s2 = subjects[j]
            
            labels_c1 = df_in_band[df_in_band['subject_id'] == s1]['label'].values
            labels_c2 = df_in_band[df_in_band['subject_id'] == s2]['label'].values
            
            # Require at least 20 samples in each condition to be meaningful (windows are 60s, overlapping)
            if len(labels_c1) < 20 or len(labels_c2) < 20:
                continue
                
            jsd, p_val, eff_z = permutation_test(labels_c1, labels_c2, all_classes, n_perm=500)
            
            results.append({
                "state": b_cfg['name'],
                "subject_1": s1,
                "subject_2": s2,
                "n1": len(labels_c1),
                "n2": len(labels_c2),
                "jsd": jsd,
                "p_value": p_val,
                "effect_z": eff_z
            })

# Apply FDR correction
if not results:
    print("No sufficient data found in any condition.")
    sys.exit(0)

df_res = pd.DataFrame(results)
df_res['q_value'] = apply_fdr(df_res['p_value'].tolist(), alpha=0.05)

# Sort by effect_z
df_res = df_res.sort_values(by="effect_z", ascending=False)

# Significant ones (divergence)
sig_breaking = df_res[df_res['q_value'] < 0.05].head(5)
print("\n--- Top 5 Invariance Breaking ---")
for idx, row in sig_breaking.iterrows():
    print(f"State: {row['state']} | {row['subject_1']} vs {row['subject_2']}")
    print(f"  JSD={row['jsd']:.3f}, p={row['p_value']:.4f}, q={row['q_value']:.4f}, z={row['effect_z']:.1f}")
    
# Least significant ones (matches)
sig_match = df_res[(df_res['q_value'] > 0.5) & (df_res['jsd'] < 0.1)].sort_values('jsd').head(5)
print("\n--- Top 5 Invariance Matches ---")
for idx, row in sig_match.iterrows():
    print(f"State: {row['state']} | {row['subject_1']} vs {row['subject_2']}")
    print(f"  JSD={row['jsd']:.3f}, p={row['p_value']:.4f}, q={row['q_value']:.4f}, z={row['effect_z']:.1f}")

df_res.to_csv("advanced_divergence_results.csv", index=False)
