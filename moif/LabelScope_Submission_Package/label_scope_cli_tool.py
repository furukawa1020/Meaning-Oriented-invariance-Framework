import pandas as pd
import numpy as np
import json
from pathlib import Path
from sklearn.neighbors import NearestNeighbors
from scipy.stats import pearsonr

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, (bool, np.bool_)): return bool(obj)
        return super(NpEncoder, self).default(obj)

def run_label_scope_audit(data_path, label_col, claim):
    df = pd.read_csv(data_path)
    
    print(f"\n--- LabelScope Audit Demonstration: {label_col} ---")
    print(f"Intended HMS Claim: {claim}\n")
    
    # 1. Resolution Audit (RA)
    uniqueness = df.groupby(['subject_id', 'Condition', 'C'])[label_col].nunique()
    is_block_level = uniqueness.max() == 1
    n_independent = 75 
    pseudo_rep_factor = len(df) / n_independent
    
    ra_level = 0 if is_block_level and n_independent < 100 and pseudo_rep_factor > 10 else 4
    
    # 2. Proxy Audit (PA)
    corr, _ = pearsonr(df[label_col].fillna(0), df['SnKeyStrokes'].fillna(0))
    pa_level = 2 if abs(corr) > 0.4 else 4
    
    # 3. Structure Audit (SA)
    features = ['HR', 'SCL', 'RMSSD']
    X = df[features].fillna(0).values
    y_bin = (df[label_col] > df[label_col].median()).astype(int).values
    
    nn = NearestNeighbors(n_neighbors=21).fit(X)
    indices = nn.kneighbors(X, return_distance=False)
    
    conflicts = [1 if (0 in y_bin[idx[1:]] and 1 in y_bin[idx[1:]]) else 0 for idx in indices]
    cr = np.mean(conflicts)
    
    y_shuf = np.random.permutation(y_bin)
    conflicts_shuf = [1 if (0 in y_shuf[idx[1:]] and 1 in y_shuf[idx[1:]]) else 0 for idx in indices]
    scr = np.mean(conflicts_shuf)
    
    sa_level = 1 if cr >= scr * 0.95 else 4

    # 4. Final Claim Assignment (The Most Restrictive Rule)
    final_level = min(ra_level, pa_level, sa_level)
    if final_level == 4 and cr > 0.35: final_level = 3
    
    # HMS Risk Descriptions
    risks = {
        0: "High risk of mis-timed interventions due to pseudo-replication.",
        1: "High risk of claiming internal-state effects from random artifacts.",
        2: "High risk of misinterpreting task volume as internal user-state.",
        3: "Inherent ambiguity detected; system-level underdetermination likely."
    }
    
    print(f"[RA] Resolution Audit: Level {ra_level} (Factor: {pseudo_rep_factor:.1f})")
    print(f"[PA] Proxy Audit: Level {pa_level} (Confounding Index: {abs(corr):.3f})")
    print(f"[SA] Structure Audit: Level {sa_level} (Conflict Rate: {cr:.3f})")
    print(f"\n>>> FINAL CLAIM LEVEL: Level {final_level}")
    print(f">>> HMS RISK: {risks.get(final_level, 'No high risks detected in the tested pipeline.')}")
    
    return {"label": label_col, "final_level": final_level}

if __name__ == "__main__":
    data_file = r"C:\Projects\Meaning-Oriented invariance Framework\moif\data\swell\swell_joined_minute_block_table.csv"
    results = []
    results.append(run_label_scope_audit(data_file, "MentalEffort", "Cognitive workload monitoring"))
    results.append(run_label_scope_audit(data_file, "error_rate", "Performance impairment detection"))
    
    with open("label_scope_final_demo_results.json", "w") as f:
        json.dump(results, f, indent=2, cls=NpEncoder)
