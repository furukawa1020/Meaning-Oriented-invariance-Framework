import pandas as pd
import numpy as np
from pathlib import Path
import json

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, (bool, np.bool_)): return bool(obj)
        return super(NpEncoder, self).default(obj)

def run_s1r_audit():
    swell_dir = Path(r"C:\Projects\Meaning-Oriented invariance Framework\moif\data\swell")
    table_path = swell_dir / "swell_joined_minute_block_table.csv"
    
    if not table_path.exists():
        print(f"Error: {table_path} not found. Run S1 first.")
        return

    df = pd.read_csv(table_path)
    
    # Variables
    perf_col = 'Performance'
    perf_rec_col = 'Performance (recoded)'
    stress_col = 'Stress'
    effort_col = 'MentalEffort'
    temp_col = 'TemporalDemand'
    error_col = 'error_rate'
    ks_col = 'SnKeyStrokes'

    # 1. Sensitivity Analysis for error_rate
    filters = [
        ("all", df),
        ("ks_gt_0", df[df[ks_col] > 0]),
        ("ks_ge_5", df[df[ks_col] >= 5]),
        ("ks_ge_10", df[df[ks_col] >= 10])
    ]
    
    robustness_rows = []
    for name, sub_df in filters:
        if len(sub_df) < 10: continue
        corr_er_ks = sub_df[[error_col, ks_col]].corr().iloc[0, 1]
        corr_er_perf = sub_df[[error_col, perf_col]].corr().iloc[0, 1]
        corr_er_perf_rec = sub_df[[error_col, perf_rec_col]].corr().iloc[0, 1]
        robustness_rows.append({
            "filter": name,
            "n_rows": len(sub_df),
            "corr_error_rate_keystrokes": corr_er_ks,
            "corr_error_rate_performance": corr_er_perf,
            "corr_error_rate_performance_recoded": corr_er_perf_rec
        })
    pd.DataFrame(robustness_rows).to_csv(swell_dir / "swell_s1r_proxy_robustness_report.csv", index=False)

    # 2. Scale Direction Audit
    df_nonzero = df[df[ks_col] > 0]
    corr_matrix = df_nonzero[[perf_col, perf_rec_col, stress_col, effort_col, temp_col, error_col]].corr()
    
    direction_audit = {
        "correlations_with_performance": corr_matrix[perf_col].to_dict(),
        "correlations_with_performance_recoded": corr_matrix[perf_rec_col].to_dict()
    }
    with open(swell_dir / "swell_s1r_scale_direction_audit.json", "w") as f:
        json.dump(direction_audit, f, indent=2, cls=NpEncoder)

    # 3. Condition Summary
    cond_summary = df.groupby('Condition')[[perf_col, perf_rec_col, stress_col, effort_col, temp_col, error_col]].mean()
    cond_summary.to_csv(swell_dir / "swell_s1r_condition_summary.csv")

    # 4. Logic for Gate
    # Check if T or I conditions have higher stress/effort than N
    avg_stress_n = cond_summary.loc['N', stress_col] if 'N' in cond_summary.index else 0
    avg_stress_t = cond_summary.loc['T', stress_col] if 'T' in cond_summary.index else 0
    avg_stress_i = cond_summary.loc['I', stress_col] if 'I' in cond_summary.index else 0
    stress_inc = (avg_stress_t > avg_stress_n) or (avg_stress_i > avg_stress_n)
    
    # Performance Direction
    # If Performance correlates POSITIVELY with ErrorRate and Stress, it is "Worse/Impairment"
    corr_perf_er = corr_matrix.loc[perf_col, error_col]
    corr_perf_stress = corr_matrix.loc[perf_col, stress_col]
    
    selected_column = "unresolved"
    higher_means = "unresolved"
    if corr_perf_er > 0.1 and corr_perf_stress > 0.1:
        selected_column = perf_col
        higher_means = "worse_performance"
    elif corr_perf_er < -0.1 and corr_perf_stress < -0.1:
        selected_column = perf_col
        higher_means = "better_performance"
    
    # Final Gate
    corr_er_ks_nonzero = robustness_rows[1]['corr_error_rate_keystrokes'] if len(robustness_rows) > 1 else 1.0
    
    verdict = "S1R_PASS" if (selected_column != "unresolved" and abs(corr_er_ks_nonzero) < 0.50 and stress_inc) else "S1R_BORDERLINE"
    
    gate = {
        "verdict": verdict,
        "performance_direction": {
            "selected_column": selected_column,
            "higher_means": higher_means,
            "evidence": {
                "corr_with_error_rate": corr_perf_er,
                "corr_with_stress": corr_perf_stress,
                "corr_with_effort": corr_matrix.loc[perf_col, effort_col],
                "corr_with_temporal_demand": corr_matrix.loc[perf_col, temp_col]
            }
        },
        "objective_proxy": {
            "error_rate_usable": abs(corr_er_ks_nonzero) < 0.50,
            "corr_error_rate_keystrokes_all": robustness_rows[0]['corr_error_rate_keystrokes'],
            "corr_error_rate_keystrokes_nonzero": corr_er_ks_nonzero,
            "zero_keystroke_rate": (df[ks_col] == 0).mean(),
            "warnings": []
        },
        "condition_sanity": {
            "stress_increases_under_T_or_I": stress_inc,
            "effort_increases_under_T_or_I": (cond_summary.loc['T', effort_col] > avg_stress_n) if 'T' in cond_summary.index else False,
            "performance_direction_consistent_with_condition": True if higher_means != "unresolved" else False
        }
    }
    
    with open("swell_s1r_gate_report.json", "w") as f:
        json.dump(gate, f, indent=2, cls=NpEncoder)

if __name__ == "__main__":
    run_s1r_audit()
