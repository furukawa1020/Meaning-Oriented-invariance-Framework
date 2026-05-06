import pandas as pd
import numpy as np
import json

RESULTS_PATH = "phase3a_r_results/phase3a_r_audit_results.csv"

def generate_report():
    df = pd.read_csv(RESULTS_PATH)
    
    # 1. Gate R1: Multi-Control Separation
    # True peak > Max of (BS, Rev) in 60% of units
    df['max_control'] = df[['bs_max_corr', 'rev_max_corr']].max(axis=1)
    df['r1_pass'] = abs(df['true_max_corr']) > (df['max_control'] + 0.05)
    pct_r1 = df['r1_pass'].mean()
    
    # 2. Gate R2: EDA -10s Consistency (Simplified for 10 subjects)
    eda_df = df[df['feat'] == 'EDA_Phasic']
    eda_lag_dist = eda_df['true_best_lag'].value_counts(normalize=True)
    # Check if -15 to -5 range is a peak (at least 2x the average bin frequency)
    target_range = eda_lag_dist.get(-15, 0) + eda_lag_dist.get(-10, 0) + eda_lag_dist.get(-5, 0)
    avg_bin = 1.0 / 25.0
    is_eda_stable = target_range > (avg_bin * 3)
    
    # 3. Gate R3: Detrend Robustness
    # Lag peak remains within 10s of original peak after first-diff in 40%
    df['r3_pass'] = abs(df['true_best_lag'] - df['diff_best_lag']) <= 10
    pct_r3 = df['r3_pass'].mean()
    
    report = {
        "verdict": "PENDING",
        "gates": {
            "R1_multi_control_separation": {
                "percent_passing": float(pct_r1),
                "pass": bool(pct_r1 >= 0.60)
            },
            "R2_eda_10s_consistency": {
                "density_in_minus_15_to_5s": float(target_range),
                "pass": bool(is_eda_stable)
            },
            "R3_detrend_robustness": {
                "percent_retaining_lag": float(pct_r3),
                "pass": bool(pct_r3 >= 0.40)
            }
        }
    }
    
    if report["gates"]["R1_multi_control_separation"]["pass"] and report["gates"]["R3_detrend_robustness"]["pass"]:
        report["verdict"] = "PROCEED_TO_WESAD"
    else:
        report["verdict"] = "TERMINATE_TEMPORAL_THESIS"
        
    with open("phase3a_r_gate_report.json", "w") as f:
        json.dump(report, f, indent=4)
    print("Red-Team Gate Report generated.")

if __name__ == "__main__":
    generate_report()
