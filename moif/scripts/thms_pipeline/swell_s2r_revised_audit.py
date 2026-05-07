import pandas as pd
import numpy as np
from pathlib import Path
import json
from sklearn.neighbors import NearestNeighbors
from scipy.stats import entropy

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, (bool, np.bool_)): return bool(obj)
        return super(NpEncoder, self).default(obj)

def run_s2r_audit():
    swell_dir = Path(r"C:\Projects\Meaning-Oriented invariance Framework\moif\data\swell")
    table_path = swell_dir / "swell_joined_minute_block_table.csv"
    df_min = pd.read_csv(table_path)
    
    # 1. Audit Block Uniqueness for Questionnaires
    q_cols = ['Performance (recoded)', 'Stress', 'MentalEffort', 'TemporalDemand']
    uniqueness_check = df_min.groupby(['subject_id', 'Condition', 'C'])[q_cols].nunique()
    block_variability_detected = bool((uniqueness_check > 1).any().any())

    # 2. Block-level Aggregation
    df_work = df_min[df_min['Condition'].isin(['N', 'T', 'I'])].copy()
    agg_dict = {
        'HR': 'mean',
        'RMSSD': 'mean',
        'SCL': 'mean',
        'SnErrorKeys': 'sum',
        'SnKeyStrokes': 'sum',
        'Performance (recoded)': 'mean',
        'Stress': 'mean',
        'MentalEffort': 'mean',
        'TemporalDemand': 'mean'
    }
    df_block = df_work.groupby(['subject_id', 'Condition', 'C']).agg(agg_dict).reset_index()
    
    # 3. Within-participant Normalization (Work blocks only)
    for sid in df_block['subject_id'].unique():
        mask = df_block['subject_id'] == sid
        for col in ['HR', 'SCL', 'RMSSD']:
            vals = df_block.loc[mask, col]
            if len(vals) > 1 and vals.std() > 0:
                df_block.loc[mask, f'z_{col}'] = (vals - vals.mean()) / vals.std()
            else:
                df_block.loc[mask, f'z_{col}'] = 0.0

    # 4. PLI (Physiological Load Index)
    df_block['PLI'] = df_block['z_HR'] + df_block['z_SCL'] - df_block['z_RMSSD']
    
    # 5. Label Assignment (Global Percentiles)
    pli_p70 = df_block['PLI'].quantile(0.7)
    perf_p30, perf_p70 = df_block['Performance (recoded)'].quantile([0.3, 0.7])
    
    df_block['is_high_load'] = df_block['PLI'] >= pli_p70
    
    df_block['impairment_label'] = np.nan
    df_block.loc[df_block['Performance (recoded)'] >= perf_p70, 'impairment_label'] = 1 # High Impairment
    df_block.loc[df_block['Performance (recoded)'] <= perf_p30, 'impairment_label'] = 0 # Preserved Performance

    # 6. Sensitivity: Participant-wise Thresholds
    for sid in df_block['subject_id'].unique():
        mask = df_block['subject_id'] == sid
        p_vals = df_block.loc[mask, 'Performance (recoded)']
        if len(p_vals) >= 3:
            p30, p70 = p_vals.quantile([0.3, 0.7])
            df_block.loc[mask, 'impairment_label_pw'] = np.nan
            df_block.loc[mask & (df_block['Performance (recoded)'] >= p70), 'impairment_label_pw'] = 1
            df_block.loc[mask & (df_block['Performance (recoded)'] <= p30), 'impairment_label_pw'] = 0

    # 7. Neighborhood Analysis (REVISED: HIGH-LOAD ONLY)
    features = ['z_HR', 'z_SCL', 'z_RMSSD']
    df_hl_labeled = df_block[df_block['is_high_load'] & df_block['impairment_label'].notnull()].copy()
    
    if len(df_hl_labeled) < 11:
        results = {"verdict": "S2R_FAIL_SUFFICIENCY", "msg": f"Insufficient high-load labeled blocks: {len(df_hl_labeled)}", "sufficiency": {"high_load_labeled_blocks": len(df_hl_labeled)}}
        with open("swell_s2r_gate_report.json", "w") as f: json.dump(results, f, indent=2, cls=NpEncoder)
        return

    X_hl = df_hl_labeled[features].values
    y_hl = df_hl_labeled['impairment_label'].values
    
    k_list = [5, 10, 20]
    audit_results = {}
    
    for k in k_list:
        if len(df_hl_labeled) <= k: continue
        nn = NearestNeighbors(n_neighbors=k + 1)
        nn.fit(X_hl)
        distances, indices = nn.kneighbors(X_hl)
        
        conflicts = []
        for i in range(len(df_hl_labeled)):
            neigh_idx = indices[i][1:]
            neigh_labels = y_hl[neigh_idx]
            conflicts.append(1 if (0 in neigh_labels and 1 in neigh_labels) else 0)
        
        audit_results[f"k{k}"] = {
            "conflict_rate": np.mean(conflicts),
            "n_samples": len(df_hl_labeled)
        }

    # 8. Shuffle Control (High-load Stratum ONLY)
    true_conflict_k10 = audit_results["k10"]["conflict_rate"]
    shuffled_conflicts = []
    nn_k10 = NearestNeighbors(n_neighbors=11).fit(X_hl)
    _, indices_k10 = nn_k10.kneighbors(X_hl)
    
    for _ in range(1000):
        y_hl_shuf = np.random.permutation(y_hl)
        c_shuf = []
        for i in range(len(df_hl_labeled)):
            neigh_labels = y_hl_shuf[indices_k10[i][1:]]
            c_shuf.append(1 if (0 in neigh_labels and 1 in neigh_labels) else 0)
        shuffled_conflicts.append(np.mean(c_shuf))
    
    p_value = np.mean(np.array(shuffled_conflicts) <= true_conflict_k10)

    # 9. Gate S2-1 Sufficiency Audit
    suff = {
        "total_valid_work_blocks": len(df_block),
        "high_load_all_blocks": int(sum(df_block['is_high_load'])),
        "high_load_labeled_blocks": len(df_hl_labeled),
        "high_load_high_impairment_blocks": int(sum(df_hl_labeled['impairment_label'] == 1)),
        "high_load_preserved_blocks": int(sum(df_hl_labeled['impairment_label'] == 0))
    }

    # 10. Gate Verdict
    verdict = "S2R_PASS"
    if suff['total_valid_work_blocks'] < 100 or suff['high_load_all_blocks'] < 30 or suff['high_load_high_impairment_blocks'] < 10 or suff['high_load_preserved_blocks'] < 10:
        verdict = "S2R_FAIL_SUFFICIENCY"
    elif true_conflict_k10 < 0.35:
        verdict = "S2R_FAIL_AMBIGUITY"
    elif p_value >= 0.05:
        verdict = "S2R_FAIL_RANDOM"
    
    p_high_imp_hl = suff['high_load_high_impairment_blocks'] / suff['high_load_labeled_blocks']
    if not (0.25 <= p_high_imp_hl <= 0.75):
        verdict = "S2R_FAIL_DETERMINISM"

    gate = {
        "verdict": verdict,
        "sample_sufficiency": suff,
        "high_load_ambiguity": audit_results,
        "shuffle_control": {
            "true_conflict_k10": true_conflict_k10,
            "shuffled_mean": np.mean(shuffled_conflicts),
            "p_value": p_value
        },
        "underdetermination_probability": p_high_imp_hl,
        "warnings": [
            "Performance (recoded) is treated as an operational subjective impairment proxy, not a clinical presenteeism measure.",
            "Objective error_rate is secondary due to residual typing-volume dependence.",
            "R condition is excluded from primary S2 normalization and analysis.",
            "Primary analysis is block-level; minute-level inference is not used.",
            f"Block questionnaire variability detected: {block_variability_detected}"
        ]
    }
    
    with open("swell_s2r_gate_report.json", "w") as f:
        json.dump(gate, f, indent=2, cls=NpEncoder)
    
    # Save sensitivity tables
    df_hl_labeled.to_csv(swell_dir / "swell_s2r_high_load_neighborhood_table.csv", index=False)

if __name__ == "__main__":
    run_s2r_audit()
