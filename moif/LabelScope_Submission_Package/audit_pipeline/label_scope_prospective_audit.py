import pandas as pd
import numpy as np
from pathlib import Path
import json
from sklearn.neighbors import NearestNeighbors
from scipy.stats import pearsonr

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, (bool, np.bool_)): return bool(obj)
        return super(NpEncoder, self).default(obj)

def run_prospective_audit():
    swell_dir = Path(r"C:\Projects\Meaning-Oriented invariance Framework\moif\data\swell")
    table_path = swell_dir / "swell_joined_minute_block_table.csv"
    df = pd.read_csv(table_path)
    
    # 1. Target Definition
    label_col = 'MentalEffort' # THE HELD-OUT LABEL
    df_work = df[df['Condition'].isin(['N', 'T', 'I'])].copy()
    
    # 2. Resolution Audit (RA)
    # Check uniqueness of MentalEffort per block
    uniqueness = df_work.groupby(['subject_id', 'Condition', 'C'])[label_col].nunique()
    is_block_level = uniqueness.max() == 1
    pseudo_rep_factor = len(df_work) / (len(df_work['subject_id'].unique()) * 3 * 3) # Assuming 3x3 blocks
    
    ra_verdict = "Level 0: Invalid" if is_block_level and pseudo_rep_factor > 10 else "PASS"

    # 3. Proxy Audit (PA) - Confounding with Keystrokes
    corr, _ = pearsonr(df_work[label_col].fillna(0), df_work['SnKeyStrokes'].fillna(0))
    pa_verdict = "Level 2: Confounded" if abs(corr) > 0.4 else "PASS"

    # 4. Structure Audit (SA)
    # Normalize features within-participant
    features = ['HR', 'SCL', 'RMSSD']
    for sid in df_work['subject_id'].unique():
        mask = df_work['subject_id'] == sid
        for col in features:
            vals = df_work.loc[mask, col]
            if vals.std() > 0:
                df_work.loc[mask, f'z_{col}'] = (vals - vals.mean()) / vals.std()
            else:
                df_work.loc[mask, f'z_{col}'] = 0.0
    
    z_feats = [f'z_{c}' for c in features]
    X = df_work[z_feats].values
    y = df_work[label_col].values
    
    # Simple median split for neighborhood conflict calculation
    y_bin = (y > np.median(y)).astype(int)
    
    nn = NearestNeighbors(n_neighbors=51).fit(X)
    _, indices = nn.kneighbors(X)
    
    conflicts = []
    for i in range(len(X)):
        neigh_labels = y_bin[indices[i][1:]]
        conflicts.append(1 if (0 in neigh_labels and 1 in neigh_labels) else 0)
    
    # Shuffle control
    y_shuf = np.random.permutation(y_bin)
    shuf_conflicts = []
    for i in range(len(X)):
        neigh_labels_shuf = y_shuf[indices[i][1:]]
        shuf_conflicts.append(1 if (0 in neigh_labels_shuf and 1 in neigh_labels_shuf) else 0)
        
    sa_verdict = "Level 1: Random-like" if np.mean(conflicts) >= np.mean(shuf_conflicts) * 0.95 else "PASS"

    # 5. Final Report
    claim_level = 4
    if ra_verdict.startswith("Level 0"): claim_level = 0
    elif pa_verdict.startswith("Level 2"): claim_level = 2
    elif sa_verdict.startswith("Level 1"): claim_level = 1
    
    report = {
        "label": label_col,
        "audits": {
            "RA": {"is_block_level": bool(is_block_level), "pseudo_rep_factor": pseudo_rep_factor, "verdict": ra_verdict},
            "PA": {"corr_with_keystrokes": corr, "verdict": pa_verdict},
            "SA": {"conflict_rate": np.mean(conflicts), "shuffle_mean": np.mean(shuf_conflicts), "verdict": sa_verdict}
        },
        "final_claim_level": claim_level,
        "recommendation": "DO NOT MODEL" if claim_level <= 2 else "PROCEED WITH CAUTION"
    }
    
    with open("label_scope_prospective_report.json", "w") as f:
        json.dump(report, f, indent=2, cls=NpEncoder)
    print(f"Prospective Audit for {label_col} Completed.")

if __name__ == "__main__":
    run_prospective_audit()
