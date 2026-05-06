import pandas as pd
import numpy as np
import json
import os
from scipy.stats import wilcoxon
from sklearn.utils import resample

RESULTS_PATH = "phase2f_results/phase2f_full_results.csv"

def bootstrap_ci(data, n_boot=2000):
    if len(data) < 2: return [np.nan, np.nan]
    stats = []
    for _ in range(n_boot):
        sample = resample(data)
        stats.append(np.mean(sample))
    return [float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))]

def cohens_d(group1, group2):
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_sd = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    return (np.mean(group1) - np.mean(group2)) / pooled_sd if pooled_sd > 1e-8 else 0

def run_consolidation():
    df = pd.read_csv(RESULTS_PATH)
    
    # --- 1. Ceiling Effect Audit ---
    ceiling_thresh = 0.98
    df['is_ceiling'] = df['auroc'] >= ceiling_thresh
    
    # We define ceiling per subject-model-dataset (if any normalizer hits it)
    df_pivot = df.pivot_table(index=['dataset', 'subject_id', 'model'], columns='norm', values='auroc')
    df_pivot['max_auroc'] = df_pivot.max(axis=1)
    df_pivot['is_ceiling'] = df_pivot['max_auroc'] >= ceiling_thresh
    
    # Primary comparison: covariance_calibration vs baseline_z
    df_pivot['delta'] = df_pivot['covariance_calibration'] - df_pivot['baseline_z']
    
    df_non_ceiling = df_pivot[~df_pivot['is_ceiling']].copy()
    
    ceiling_audit = {
        "total_slots": len(df_pivot),
        "ceiling_slots_auroc_ge_0_98": int(df_pivot['is_ceiling'].sum()),
        "non_ceiling_slots": len(df_non_ceiling),
        "delta_auroc_all_slots": float(df_pivot['delta'].mean()),
        "delta_auroc_non_ceiling_only": float(df_non_ceiling['delta'].mean()),
        "subject_support_all_slots": float((df_pivot['delta'] >= 0.03).mean()),
        "subject_support_non_ceiling_only": float((df_non_ceiling['delta'] >= 0.03).mean()),
        "dataset_support_non_ceiling": df_non_ceiling.groupby('dataset')['delta'].apply(lambda x: (x >= 0.03).mean()).to_dict()
    }
    with open("ceiling_effect_audit.json", "w") as f:
        json.dump(ceiling_audit, f, indent=4)

    # --- 2. Statistical Support Report ---
    stats_report = []
    for model in ['LR', 'RF', 'SVM']:
        for ds in ['WESAD', 'CASE']:
            subset = df_pivot[(df_pivot.index.get_level_values('model') == model) & 
                              (df_pivot.index.get_level_values('dataset') == ds)]
            if subset.empty: continue
            
            # Non-ceiling subset for CI/P
            sub_nc = subset[~subset['is_ceiling']]
            if len(sub_nc) < 5: 
                mean_d, ci, p, d_val = 0, [0,0], 1, 0
            else:
                mean_d = sub_nc['delta'].mean()
                ci = bootstrap_ci(sub_nc['delta'].values)
                _, p = wilcoxon(sub_nc['covariance_calibration'], sub_nc['baseline_z'])
                d_val = cohens_d(sub_nc['covariance_calibration'].values, sub_nc['baseline_z'].values)
            
            stats_report.append({
                "model": model, "dataset": ds, "n_non_ceiling": len(sub_nc),
                "mean_delta_auroc": float(mean_d),
                "bootstrap_ci_95": ci,
                "wilcoxon_p": float(p),
                "effect_size_d": float(d_val)
            })
    with open("statistical_support_report.json", "w") as f:
        json.dump(stats_report, f, indent=4)

    # --- 3. Model-Feature Interaction Audit ---
    # Interaction summary
    interaction = {}
    for entry in stats_report:
        m = entry['model']
        if m not in interaction: interaction[m] = {"delta_sum": 0, "n": 0, "pass_count": 0}
        interaction[m]["delta_sum"] += entry['mean_delta_auroc']
        interaction[m]["n"] += 1
        if entry['bootstrap_ci_95'][0] > 0: interaction[m]["pass_count"] += 1
        
    audit_interaction = {}
    for m, vals in interaction.items():
        avg_d = vals['delta_sum'] / vals['n']
        audit_interaction[m] = {
            "delta_auroc": float(avg_d),
            "pass": bool(vals['pass_count'] == vals['n'] and avg_d >= 0.03)
        }
    
    audit_interaction["interpretation"] = "model_dependent" if audit_interaction['SVM']['delta_auroc'] < 0.01 else "general"
    with open("model_feature_interaction_audit.json", "w") as f:
        json.dump(audit_interaction, f, indent=4)

    print("Phase 2G consolidation JSONs generated.")

if __name__ == "__main__":
    run_consolidation()
