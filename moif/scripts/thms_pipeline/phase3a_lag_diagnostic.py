import pandas as pd
import numpy as np
import json
import os

REPORT_PATH = "phase3a_results/case_lag_correlation_report.csv"

def generate_diagnostic():
    df = pd.read_csv(REPORT_PATH)
    
    # 1. Gate T1: Lag non-zero structure
    # Percentage of units where best lag is outside ±5s
    outside_5s = df[abs(df['best_lag']) > 5]
    pct_outside_5s = len(outside_5s) / len(df)
    
    # 2. Gate T3: Control separation
    # Percentage of units where max_corr > ctrl_max_corr + 0.05
    df['is_significant'] = abs(df['max_corr']) > (df['ctrl_max_corr'] + 0.05)
    pct_significant = df['is_significant'].mean()
    
    # 3. Median Optimal Lags by Feature/Label
    lag_summary = df.groupby(['feat', 'label'])['best_lag'].agg(['mean', 'median', 'std']).to_dict()
    
    # 4. Correlation Gain
    df['corr_gain'] = abs(df['max_corr']) - abs(df['zero_lag_corr'])
    mean_gain = df['corr_gain'].mean()
    
    report = {
        "verdict": "PENDING",
        "gates": {
            "T1_lag_non_zero_structure": {
                "percent_outside_pm_5s": float(pct_outside_5s),
                "pass": bool(pct_outside_5s >= 0.30)
            },
            "T3_control_separation": {
                "percent_exceeding_control_by_0_05": float(pct_significant),
                "pass": bool(pct_significant >= 0.60)
            }
        },
        "statistics": {
            "mean_correlation_gain": float(mean_gain),
            "lag_summary": lag_summary
        }
    }
    
    if report["gates"]["T1_lag_non_zero_structure"]["pass"] and report["gates"]["T3_control_separation"]["pass"]:
        report["verdict"] = "PROCEED_TO_PHASE_3B"
    else:
        report["verdict"] = "REVAL_OR_TERMINATE"
        
    with open("phase3a_lag_gate_report.json", "w") as f:
        json.dump(report, f, indent=4)
    print("Phase 3A Lag Gate Report generated.")

if __name__ == "__main__":
    generate_diagnostic()
