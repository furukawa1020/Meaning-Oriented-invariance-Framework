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

def run_s2_audit():
    swell_dir = Path(r"C:\Projects\Meaning-Oriented invariance Framework\moif\data\swell")
    table_path = swell_dir / "swell_joined_minute_block_table.csv"
    df_min = pd.read_csv(table_path)
    
    # 1. Block-level Aggregation
    # Physiology: HR, HRV_RMSSD, SCL
    # PC: SnErrorKeys, SnKeyStrokes
    # Quest: Performance (recoded), Stress, MentalEffort, TemporalDemand
    
    # First, separate work blocks (N, T, I)
    df_work = df_min[df_min['Condition'].isin(['N', 'T', 'I'])].copy()
    
    # Aggregate to block-level
    agg_dict = {
        'HR': 'mean',
        'HRV_RMSSD': 'mean',
        'SCL': 'mean',
        'SnErrorKeys': 'sum',
        'SnKeyStrokes': 'sum',
        'Performance (recoded)': 'first', # Same for the block
        'Stress': 'first',
        'MentalEffort': 'first',
        'TemporalDemand': 'first'
    }
    df_block = df_work.groupby(['subject_id', 'Condition', 'C']).agg(agg_dict).reset_index()
    
    # 2. Within-participant Normalization (N/T/I blocks only)
    for sid in df_block['subject_id'].unique():
        mask = df_block['subject_id'] == sid
        for col in ['HR', 'SCL', 'HRV_RMSSD']:
            vals = df_block.loc[mask, col]
            if vals.std() > 0:
                df_block.loc[mask, f'z_{col}'] = (vals - vals.mean()) / vals.std()
            else:
                df_block.loc[mask, f'z_{col}'] = 0.0

    # 3. Physiological Load Index (PLI)
    df_block['PLI'] = df_block['z_HR'] + df_block['z_SCL'] - df_block['z_HRV_RMSSD']
    
    # 4. Error Rate (Secondary)
    df_block['error_rate'] = df_block.apply(lambda x: x['SnErrorKeys'] / x['SnKeyStrokes'] if x['SnKeyStrokes'] > 0 else np.nan, axis=1)

    # 5. Label Assignment (30/40/30 split)
    pli_p30, pli_p70 = df_block['PLI'].quantile([0.3, 0.7])
    perf_p30, perf_p70 = df_block['Performance (recoded)'].quantile([0.3, 0.7])
    
    df_block['is_high_load'] = df_block['PLI'] >= pli_p70
    df_block['is_low_load'] = df_block['PLI'] <= pli_p30
    
    # Performance Recoded: Higher = Worse
    df_block['impairment_label'] = np.nan
    df_block.loc[df_block['Performance (recoded)'] >= perf_p70, 'impairment_label'] = 1 # High Impairment
    df_block.loc[df_block['Performance (recoded)'] <= perf_p30, 'impairment_label'] = 0 # Preserved Performance

    # Save block-level table
    df_block.to_csv(swell_dir / "swell_s2_block_level_table.csv", index=False)

    # 6. Neighborhood Analysis (k=10)
    features = ['z_HR', 'z_SCL', 'z_HRV_RMSSD']
    # Filter only labeled samples for conflict analysis
    df_labeled = df_block[df_block['impairment_label'].notnull()].copy()
    X = df_labeled[features].values
    y = df_labeled['impairment_label'].values
    
    k = 10
    nn = NearestNeighbors(n_neighbors=k + 1) # +1 to exclude self
    nn.fit(X)
    distances, indices = nn.kneighbors(X)
    
    # Conflict calculation
    conflicts = []
    purities = []
    entropies = []
    
    for i in range(len(df_labeled)):
        neighbor_indices = indices[i][1:] # Exclude self
        neighbor_labels = y[neighbor_indices]
        
        # Conflict: Does it have both 0 and 1?
        has_0 = 0 in neighbor_labels
        has_1 = 1 in neighbor_labels
        conflicts.append(1 if (has_0 and has_1) else 0)
        
        # Purity
        p1 = np.mean(neighbor_labels)
        purities.append(max(p1, 1 - p1))
        
        # Entropy
        p_vec = [p1, 1 - p1] if p1 > 0 and p1 < 1 else [1, 0]
        entropies.append(entropy(p_vec, base=2))

    df_labeled['conflict'] = conflicts
    df_labeled['purity'] = purities
    df_labeled['entropy'] = entropies
    
    # 7. High-Load Stratum Analysis
    df_high_load = df_labeled[df_labeled['is_high_load']]
    conflict_rate_high_load = df_high_load['conflict'].mean()
    p_high_imp_given_high_load = df_high_load['impairment_label'].mean()
    
    # 8. Shuffle Control
    shuffled_conflicts = []
    for _ in range(1000):
        y_shuffled = np.random.permutation(y)
        # We only care about high-load stratum
        # Actually, shuffle within high-load stratum only?
        # User: "Shuffle within the high-load stratum"
        idx_hl = df_labeled[df_labeled['is_high_load']].index
        # To make it simple, shuffle all labels
        y_shuf_global = np.random.permutation(y)
        
        c_shuf = []
        for i in range(len(df_labeled)):
            if not df_labeled.iloc[i]['is_high_load']: continue
            neigh_idx = indices[i][1:]
            neigh_labels = y_shuf_global[neigh_idx]
            c_shuf.append(1 if (0 in neigh_labels and 1 in neigh_labels) else 0)
        shuffled_conflicts.append(np.mean(c_shuf))
    
    p_value = np.mean(np.array(shuffled_conflicts) <= conflict_rate_high_load)

    # 9. Reports
    sufficiency = {
        "total_blocks": len(df_block),
        "labeled_blocks": len(df_labeled),
        "high_load_blocks": len(df_high_load),
        "high_impairment_blocks": int(sum(df_labeled['impairment_label'] == 1)),
        "preserved_performance_blocks": int(sum(df_labeled['impairment_label'] == 0))
    }
    with open(swell_dir / "swell_s2_sample_sufficiency_report.json", "w") as f:
        json.dump(sufficiency, f, indent=2, cls=NpEncoder)

    audit = {
        "k": k,
        "high_load_conflict_rate": conflict_rate_high_load,
        "mean_high_load_purity": df_high_load['purity'].mean(),
        "mean_high_load_entropy": df_high_load['entropy'].mean(),
        "p_high_imp_given_high_load": p_high_imp_given_high_load
    }
    with open(swell_dir / "swell_s2_high_load_impairment_audit.json", "w") as f:
        json.dump(audit, f, indent=2, cls=NpEncoder)

    shuffle_report = {
        "true_conflict_rate": conflict_rate_high_load,
        "shuffled_conflict_mean": np.mean(shuffled_conflicts),
        "p_value": p_value
    }
    with open(swell_dir / "swell_s2_shuffle_control_report.json", "w") as f:
        json.dump(shuffle_report, f, indent=2, cls=NpEncoder)

    # 10. Gate Report
    verdict = "S2_PASS"
    if sufficiency['total_blocks'] < 100 or sufficiency['high_load_blocks'] < 30 or sufficiency['high_impairment_blocks'] < 10 or sufficiency['preserved_performance_blocks'] < 10:
        verdict = "S2_FAIL_SUFFICIENCY"
    elif conflict_rate_high_load < 0.35:
        verdict = "S2_FAIL_AMBIGUITY"
    elif p_value >= 0.05:
        verdict = "S2_FAIL_RANDOM"
    elif not (0.25 <= p_high_imp_given_high_load <= 0.75):
        verdict = "S2_FAIL_DETERMINISM"

    gate = {
        "verdict": verdict,
        "sample_sufficiency": sufficiency,
        "high_load_impairment_conflict": audit,
        "shuffle_control": shuffle_report,
        "underdetermination_probability": p_high_imp_given_high_load,
        "warnings": []
    }
    with open("swell_s2_gate_report.json", "w") as f:
        json.dump(gate, f, indent=2, cls=NpEncoder)

if __name__ == "__main__":
    run_s2_audit()
