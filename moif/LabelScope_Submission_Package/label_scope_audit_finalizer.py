import json
from pathlib import Path

def generate_final_audit():
    scratch_dir = Path(".")
    
    # Reports to collect
    report_paths = {
        "SWELL_S2": scratch_dir / "swell_s2r_gate_report.json",
        "SWELL_S3": scratch_dir / "swell_s3_gate_report.json"
    }
    
    final_audit = {
        "framework_version": "LabelScope-v1.0",
        "audit_target": "SWELL-KW Multi-Modal Dataset and Supplementary Studies",
        "diagnostics": [],
        "cross_dataset_insights": [
            {
                "target": "CASE (Arousal/Valence)",
                "audit_module": "Structure Audit",
                "finding": "Ambiguity exists but lacks significant contrast to discriminative axes (Arousal).",
                "claim_level": "Level 1: Dataset-Specific Observation"
            },
            {
                "target": "WESAD/CASE (Normalization)",
                "audit_module": "Claim Audit",
                "finding": "Apparent state effects can be confounded by dataset offsets and baseline anomalies.",
                "claim_level": "Level 1: Dataset-Specific Observation"
            },
            {
                "target": "CASE/WESAD (Lag)",
                "audit_module": "Structure/Claim Audit",
                "finding": "Apparent lag is indistinguishable from low-frequency drift and autocorrelation artifacts.",
                "claim_level": "Level 1: Dataset-Specific Observation"
            }
        ]
    }
    
    # 1. Audit SWELL S2
    try:
        with open(report_paths["SWELL_S2"], "r") as f:
            s2 = json.load(f)
        final_audit["diagnostics"].append({
            "label": "Performance (recoded)",
            "audit_module": "Resolution Audit",
            "failure_mode": "Resolution Mismatch / Sample Insufficiency",
            "findings": s2.get("sample_sufficiency", {}),
            "verdict": s2.get("verdict", "FAIL"),
            "claim_level": "Level 0: Invalid Candidate",
            "recommendation": "Questionnaire resolution is insufficient for minute-level physiology modeling in this context."
        })
    except: pass

    # 2. Audit SWELL S3
    try:
        with open(report_paths["SWELL_S3"], "r") as f:
            s3 = json.load(f)
        final_audit["diagnostics"].append({
            "label": "Residual Objective Error",
            "audit_module": "Structure Audit",
            "failure_mode": "Random-like Label Structure",
            "findings": s3.get("shuffle_control", {}),
            "verdict": s3.get("verdict", "FAIL"),
            "claim_level": "Level 1: Dataset-Specific Observation",
            "recommendation": "No detectable structure was found in physiological neighborhood under current audit settings."
        })
    except: pass

    # 3. Add Proxy Confounding (SWELL S1 Error Rate)
    final_audit["diagnostics"].append({
        "label": "error_rate",
        "audit_module": "Proxy Audit",
        "failure_mode": "Proxy Confounding",
        "findings": {"corr_with_keystrokes": 0.604},
        "verdict": "S1R_BORDERLINE",
        "claim_level": "Level 2: Behavior-Specific Observation",
        "recommendation": "Label is partially confounded with task volume and should not be used as a pure internal-state proxy."
    })

    # Save final report
    with open("label_scope_final_audit_report.json", "w") as f:
        json.dump(final_audit, f, indent=2)
    
    print("LabelScope Final Audit Report Generated (Level 0-2 focused).")

if __name__ == "__main__":
    generate_final_audit()
