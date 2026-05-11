import pandas as pd
import numpy as np
from pathlib import Path
import json
import statsmodels.formula.api as smf
from sklearn.neighbors import NearestNeighbors
from scipy.stats import entropy

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, (bool, np.bool_)): return bool(obj)
        return super(NpEncoder, self).default(obj)

def run_s3_audit():
    swell_dir = Path(r"C:\Projects\Meaning-Oriented invariance Framework\moif\data\swell")
    table_path = swell_dir / "swell_joined_minute_block_table.csv"
    df = pd.read_csv(table_path)
    
    # 1. Filter Work Blocks
    df_work = df[df['Condition'].isin(['N', 'T', 'I'])].copy()
    df_work = df_work.rename(columns={'C': 'block_id'})
    
    # Clean numeric columns
    for col in ['SnErrorKeys', 'SnKeyStrokes', 'HR', 'SCL', 'RMSSD']:
        df_work[col] = pd.to_numeric(df_work[col], errors='coerce').fillna(0)

    # 2. Within-participant PLI Normalization
    for sid in df_work['subject_id'].unique():
        mask = df_work['subject_id'] == sid
        for col in ['HR', 'SCL', 'RMSSD']:
            vals = df_work.loc[mask, col]
            if len(vals) > 1 and vals.std() > 0:
                df_work.loc[mask, f'z_{col}'] = (vals - vals.mean()) / vals.std()
            else:
                df_work.loc[mask, f'z_{col}'] = 0.0
    df_work['PLI'] = df_work['z_HR'] + df_work['z_SCL'] - df_work['z_RMSSD']

    # 3. Residual Error Model
    # SnErrorKeys ~ SnKeyStrokes + C(subject_id) + C(Condition)
    # We use a simple linear model for residuals
    model = smf.ols('SnErrorKeys ~ SnKeyStrokes + C(subject_id) + C(Condition)', data=df_work).fit()
    df_work['residual_error'] = model.resid
    
    # Validate Residuals
    corr_resid_ks = df_work[['residual_error', 'SnKeyStrokes']].corr().iloc[0, 1]
    
    # 4. Labeling Residual Error (Top/Bottom 30%)
    res_p30, res_p70 = df_work['residual_error'].quantile([0.3, 0.7])
    pli_p70 = df_work['PLI'].quantile(0.7)
    
    df_work['is_high_load'] = df_work['PLI'] >= pli_p70
    df_work['error_label'] = np.nan
    df_work.loc[df_work['residual_error'] >= res_p70, 'error_label'] = 1 # High Residual Error
    df_work.loc[df_work['residual_error'] <= res_p30, 'error_label'] = 0 # Low Residual Error

    # 5. Neighborhood Analysis (k=50, 100)
    features = ['z_HR', 'z_SCL', 'z_RMSSD']
    df_labeled = df_work[df_work['error_label'].notnull()].copy()
    X = df_labeled[features].values
    y = df_labeled['error_label'].values
    
    k_list = [50, 100]
    audit_results = {}
    
    for k in k_list:
        if len(df_labeled) <= k: continue
        nn = NearestNeighbors(n_neighbors=k + 1)
        nn.fit(X)
        distances, indices = nn.kneighbors(X)
        
        # Only analyze high-load stratum queries
        is_hl_mask = df_labeled['is_high_load'].values
        conflicts = []
        for i in range(len(df_labeled)):
            if not is_hl_mask[i]: continue
            neigh_idx = indices[i][1:]
            neigh_labels = y[neigh_idx]
            conflicts.append(1 if (0 in neigh_labels and 1 in neigh_labels) else 0)
        
        audit_results[f"k{k}"] = {
            "conflict_rate": np.mean(conflicts) if conflicts else 0.0,
            "n_high_load_queries": len(conflicts)
        }

    # 6. Shuffle Control (k=100)
    true_conflict_k100 = audit_results["k100"]["conflict_rate"] if "k100" in audit_results else 0.0
    shuffled_conflicts = []
    
    # Pre-calculate indices for high-load only
    hl_indices = np.where(df_labeled['is_high_load'])[0]
    
    if "k100" in audit_results:
        nn_k100 = NearestNeighbors(n_neighbors=101).fit(X)
        _, indices_k100 = nn_k100.kneighbors(X[hl_indices])
        
        for _ in range(200): # Faster shuffle for minute-level
            y_shuf = np.random.permutation(y)
            c_shuf = []
            for i in range(len(hl_indices)):
                neigh_labels = y_shuf[indices_k100[i][1:]]
                c_shuf.append(1 if (0 in neigh_labels and 1 in neigh_labels) else 0)
            shuffled_conflicts.append(np.mean(c_shuf))
        
        p_value = np.mean(np.array(shuffled_conflicts) <= true_conflict_k100)
    else:
        p_value = 1.0

    # 7. Gate Report
    p_high_err_hl = df_labeled[df_labeled['is_high_load']]['error_label'].mean()
    
    verdict = "S3_PASS"
    if len(df_work) < 2000: verdict = "S3_FAIL_SIZE"
    elif abs(corr_resid_ks) > 0.1: verdict = "S3_FAIL_MODEL"
    elif true_conflict_k100 < 0.35: verdict = "S3_FAIL_AMBIGUITY"
    elif p_value >= 0.05: verdict = "S3_FAIL_RANDOM"
    elif not (0.25 <= p_high_err_hl <= 0.75): verdict = "S3_FAIL_DETERMINISM"

    gate = {
        "verdict": verdict,
        "model_validity": {
            "corr_residual_keystrokes": corr_resid_ks,
            "r_squared": model.rsquared
        },
        "sample_sufficiency": {
            "total_minutes": len(df_work),
            "high_load_labeled_minutes": int(sum(df_labeled['is_high_load']))
        },
        "high_load_ambiguity": audit_results,
        "shuffle_control": {
            "true_conflict_k100": true_conflict_k100,
            "p_value": p_value
        },
        "underdetermination_probability": p_high_err_hl,
        "warnings": [
            "Residual error is a statistical construct, not a direct measure of impairment.",
            "Clustered bootstrap not fully implemented in this script; manual review suggested."
        ]
    }
    
    with open("swell_s3_gate_report.json", "w") as f:
        json.dump(gate, f, indent=2, cls=NpEncoder)
    
    # Save residual data for visualization
    df_work.to_csv(swell_dir / "swell_s3_residual_table.csv", index=False)

if __name__ == "__main__":
    run_s3_audit()
