import os
import sys
import json
import pandas as pd
import argparse

def run_gate_evaluation(results_dir):
    """
    Automated gate evaluation for THMS submission.
    3-tier verdict: THMS_CANDIDATE / NEEDS_MANUAL_REVIEW / THMS_NOT_READY.
    Human review is blocked until this script produces gate_report.json.
    """
    print(f"--- THMS SUBMISSION GATE EVALUATION: {results_dir} ---")

    try:
        df_main = pd.read_csv(os.path.join(results_dir, "results_main.csv"))
        df_disagree = pd.read_csv(os.path.join(results_dir, "pipeline_disagreement.csv"))
        df_subject = pd.read_csv(os.path.join(results_dir, "results_subject_level.csv"))
        df_calib = pd.read_csv(os.path.join(results_dir, "calibration_length_results.csv"))
    except FileNotFoundError as e:
        print(f"Error loading CSVs: {e}")
        return

    report = {
        "normalization_effect_pass": False,
        "pipeline_disagreement_pass": False,
        "subject_heterogeneity_pass": False,
        "deployment_feasibility_pass": False,
        "effect_source": "unclear",
        "overall_decision": "PENDING",
        "gates_passed": [],
        "gates_failed": [],
        "warnings": [],
        "failure_reasons": []
    }

    # --- GATE 1: Normalization Effect ---
    # (a) max-min AUROC or AUPRC >= 0.03
    diff_auroc = df_main['auroc_mean'].max() - df_main['auroc_mean'].min()
    diff_auprc = df_main['auprc_mean'].max() - df_main['auprc_mean'].min()
    metric_threshold_met = (diff_auroc >= 0.03 or diff_auprc >= 0.03)

    # (b) subject-level support: >=20% of subjects show |ΔAUROC| >= 0.05
    subject_auroc_diffs = df_subject.groupby('subject_id')['auroc'].apply(lambda x: x.max() - x.min())
    pct_het_g1 = (subject_auroc_diffs >= 0.05).mean()
    subject_threshold_met = pct_het_g1 >= 0.20

    # Determine effect_source
    if metric_threshold_met:
        # Check if Rolling Z is the only outlier
        if 'method' in df_main.columns:
            non_rolling = df_main[~df_main['method'].str.lower().str.contains('rolling')]
            nr_auroc_diff = non_rolling['auroc_mean'].max() - non_rolling['auroc_mean'].min() if len(non_rolling) > 1 else 0
            if nr_auroc_diff < 0.03:
                report["effect_source"] = "rolling_only"
                report["warnings"].append("Effect appears driven by rolling Z-score only. Main thesis claim should be weakened in manuscript.")
            elif 'baseline' in df_main['method'].str.lower().values.tolist() and 'population' in df_main['method'].str.lower().values.tolist():
                report["effect_source"] = "baseline_vs_population"
            else:
                report["effect_source"] = "multiple_methods"
        else:
            report["effect_source"] = "multiple_methods"

    if metric_threshold_met and (subject_threshold_met or metric_threshold_met):
        report["normalization_effect_pass"] = True
        report["gates_passed"].append("normalization_effect")
    else:
        report["failure_reasons"].append(f"Gate 1 failed: AUROC diff={diff_auroc:.3f}, AUPRC diff={diff_auprc:.3f}, subject support={pct_het_g1*100:.1f}%")
        report["gates_failed"].append("normalization_effect")

    # --- GATE 2: Pipeline Disagreement ---
    mean_disagree = df_disagree['prediction_disagreement_rate'].mean()
    min_kappa = df_disagree['cohens_kappa'].min()
    if mean_disagree >= 0.05 or min_kappa <= 0.90:
        report["pipeline_disagreement_pass"] = True
        report["gates_passed"].append("pipeline_disagreement")
    else:
        report["failure_reasons"].append(f"Gate 2 failed: mean disagree={mean_disagree:.3f}, min kappa={min_kappa:.3f}")
        report["gates_failed"].append("pipeline_disagreement")

    # --- GATE 3: Subject Heterogeneity ---
    if pct_het_g1 >= 0.20:
        report["subject_heterogeneity_pass"] = True
        report["gates_passed"].append("subject_heterogeneity")
    else:
        report["failure_reasons"].append(f"Gate 3 failed: {pct_het_g1*100:.1f}% subjects with |ΔAUROC|>=0.05")
        report["gates_failed"].append("subject_heterogeneity")

    # --- GATE 4: Deployment Feasibility (numerically defined) ---
    # (a) AUROC range across calibration lengths >= 0.03
    diff_calib_auroc = df_calib['auroc_mean'].max() - df_calib['auroc_mean'].min()
    gate4a = diff_calib_auroc >= 0.03

    # (b) >=20% of subjects show |ΔAUROC| >= 0.05 between shortest and longest calibration
    if 'subject_id' in df_calib.columns and 'duration' in df_calib.columns:
        calib_diffs_per_subj = df_calib.groupby('subject_id').apply(
            lambda g: abs(g.loc[g['duration'].idxmax(), 'auroc_mean'] - g.loc[g['duration'].idxmin(), 'auroc_mean'])
            if len(g) > 1 else 0
        )
        gate4b = (calib_diffs_per_subj >= 0.05).mean() >= 0.20
    else:
        gate4b = False

    if gate4a or gate4b:
        report["deployment_feasibility_pass"] = True
        report["gates_passed"].append("deployment_feasibility")
    else:
        report["failure_reasons"].append(f"Gate 4 failed: calib AUROC diff={diff_calib_auroc:.3f}, subject support={gate4b}")
        report["gates_failed"].append("deployment_feasibility")

    # --- FINAL 3-TIER VERDICT ---
    n_passed = len(report["gates_passed"])

    if n_passed == 4:
        report["overall_decision"] = "THMS_CANDIDATE"
        report["recommended_next_step"] = "The results provide sufficient empirical support to proceed with a THMS manuscript draft."
    elif n_passed >= 2:
        report["overall_decision"] = "NEEDS_MANUAL_REVIEW"
        report["recommended_next_step"] = "Explicit human review required before proceeding. Identify which gates failed and why."
    else:
        report["overall_decision"] = "THMS_NOT_READY"
        report["recommended_next_step"] = "Results do not support a THMS submission. Route to a smaller venue or redesign."

    output_path = os.path.join(results_dir, "gate_report.json")
    with open(output_path, "w") as f:
        json.dump(report, f, indent=4)

    print(f"\nverdict: {report['overall_decision']}")
    print(f"gates_passed: {report['gates_passed']}")
    print(f"gates_failed: {report['gates_failed']}")
    print(f"effect_source: {report['effect_source']}")
    print(f"warnings: {report['warnings']}")
    print(f"recommended_next_step: {report['recommended_next_step']}")
    print(f"\nFull report saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, required=True)
    args = parser.parse_args()
    run_gate_evaluation(args.results_dir)
