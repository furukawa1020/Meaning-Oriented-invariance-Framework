import pandas as pd
import os
import json

RESULTS_DIR = "phase2c_results"

def generate():
    # 1. Normalization Effect
    df_main = pd.read_csv(os.path.join(RESULTS_DIR, "results_main.csv"))
    max_min_auroc = df_main['auroc_mean'].max() - df_main['auroc_mean'].min()
    max_min_auprc = df_main['auprc_mean'].max() - df_main['auprc_mean'].min()
    
    datasets_effect = []
    for ds in df_main['dataset'].unique():
        df_ds = df_main[df_main['dataset'] == ds]
        if (df_ds['auroc_mean'].max() - df_ds['auroc_mean'].min() >= 0.03 or 
            df_ds['auprc_mean'].max() - df_ds['auprc_mean'].min() >= 0.03):
            datasets_effect.append(ds)

    # 2. Pipeline Disagreement
    df_disagree = pd.read_csv(os.path.join(RESULTS_DIR, "pipeline_disagreement.csv"))
    mean_disagree = df_disagree['prediction_disagreement_rate'].mean()
    min_kappa = df_disagree['cohens_kappa'].min()
    strongest_pair = df_disagree.loc[df_disagree['prediction_disagreement_rate'].idxmax(), 'method_pair']

    # 3. Subject Heterogeneity (Gate 3)
    df_subj = pd.read_csv(os.path.join(RESULTS_DIR, "results_subject_level.csv"))
    subj_diffs = df_subj.groupby(['dataset', 'subject_id'])['auroc'].apply(lambda x: x.max() - x.min())
    pct_het = (subj_diffs >= 0.05).mean()
    
    failure_type = "hard_fail"
    if pct_het >= 0.15: failure_type = "near_miss"
    elif pct_het >= 0.08: failure_type = "moderate_fail"

    # 4. Deployment Feasibility
    df_calib = pd.read_csv(os.path.join(RESULTS_DIR, "calibration_length_results.csv"))
    calib_range = df_calib['auroc_mean'].max() - df_calib['auroc_mean'].min()
    
    # Calculate % subjects short vs long delta >= 0.05
    # Longest usually 300, shortest 30 or 60
    calib_subj_diffs = []
    for (ds, sid), group in df_calib.groupby(['dataset', 'subject_id']):
        if len(group) > 1:
            longest = group.loc[group['duration'].idxmax(), 'auroc_mean']
            shortest = group.loc[group['duration'].idxmin(), 'auroc_mean']
            calib_subj_diffs.append(abs(longest - shortest))
    pct_calib_het = (pd.Series(calib_subj_diffs) >= 0.05).mean() if calib_subj_diffs else 0.0

    diagnostics = {
        "normalization_effect": {
            "max_min_auroc": float(max_min_auroc),
            "max_min_auprc": float(max_min_auprc),
            "dataset_where_effect_observed": datasets_effect,
            "effect_source": "multiple_methods" # Hardcoded based on gate_report.json
        },
        "pipeline_disagreement": {
            "mean_prediction_disagreement_rate": float(mean_disagree),
            "minimum_cohens_kappa": float(min_kappa),
            "strongest_method_pair": strongest_pair
        },
        "subject_heterogeneity": {
            "percent_subjects_delta_auroc_ge_0_05": float(pct_het),
            "threshold": 0.20,
            "failure_type": failure_type
        },
        "deployment_feasibility": {
            "calibration_length_auroc_range": float(calib_range),
            "percent_subjects_short_vs_long_delta_ge_0_05": float(pct_calib_het)
        }
    }

    with open("gate_diagnostics_minimal.json", "w") as f:
        json.dump(diagnostics, f, indent=4)
    print("gate_diagnostics_minimal.json generated.")

if __name__ == "__main__":
    generate()
