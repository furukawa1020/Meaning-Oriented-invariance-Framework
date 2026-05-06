import os
import sys
import json
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
import datetime

# Strict guard enforcing Phase 2A
if os.environ.get("ALLOW_REAL_DATA_PHASE_2A_INGESTION") != "1":
    print("FATAL: Real data ingestion is locked before Phase 2A authorization.")
    print("Run with: ALLOW_REAL_DATA_PHASE_2A_INGESTION=1 python phase2a_ingestion_audit.py")
    sys.exit(1)

# WE STRICTLY DO NOT IMPORT NORMALIZERS, CLASSIFIERS, OR EXTRACT_INSTANTANEOUS_FEATURES HERE

WESAD_DIR = Path("C:/Projects/Meaning-Oriented invariance Framework/moif/data/wesad")
CASE_DIR = Path("C:/Projects/Meaning-Oriented invariance Framework/moif/data/case")

def write_access_log(dataset, subject_id, status):
    log_entry = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "dataset": dataset,
        "subject_id": subject_id,
        "operation": "load_metadata_only",
        "performance_metric_computed": False,
        "normalization_applied": False,
        "classifier_invoked": False,
        "status": status
    }
    with open("real_data_access_log.jsonl", "a") as f:
        f.write(json.dumps(log_entry) + "\n")

def audit_wesad():
    print("Auditing WESAD...")
    records = []
    
    # Check if dir exists
    if not WESAD_DIR.exists():
        print(f"Warning: {WESAD_DIR} not found. Returning empty list.")
        return records

    pkl_files = list(WESAD_DIR.rglob("*.pkl"))
    for p_path in pkl_files:
        try:
            with open(p_path, 'rb') as f:
                data = pickle.load(f, encoding='latin1')
                
            subj_id = data['subject']
            # labels are at 700Hz
            lbls_700hz = data['label'].flatten()
            ecg_chest = data['signal']['chest']['ECG'].flatten()
            eda_wrist = data['signal']['wrist']['EDA'].flatten()
            
            # Map labels
            # 1: baseline, 2: stress
            n_raw = len(lbls_700hz)
            n_baseline = np.sum(lbls_700hz == 1)
            n_active = np.sum(lbls_700hz == 2)
            
            # Assuming 50/50 temporal split (first half of arrays vs second half)
            # This is a rough estimation of the split sizes for the audit
            idx_baseline = np.where(lbls_700hz == 1)[0]
            idx_active = np.where(lbls_700hz == 2)[0]
            
            records.append({
                "dataset": "WESAD",
                "subject_id": subj_id,
                "n_raw_samples_700hz": n_raw,
                "n_baseline_samples": n_baseline,
                "n_active_samples": n_active,
                "baseline_duration_sec": n_baseline / 700.0,
                "active_duration_sec": n_active / 700.0,
                "available_channels": "chest:ECG, wrist:EDA",
                "status": "SUCCESS",
                "failure_reason": ""
            })
            write_access_log("WESAD", subj_id, "SUCCESS")
        except Exception as e:
            records.append({
                "dataset": "WESAD",
                "subject_id": p_path.stem,
                "status": "FAILED",
                "failure_reason": str(e)
            })
            write_access_log("WESAD", p_path.stem, "FAILED")
            
    return records

def audit_case():
    print("Auditing CASE...")
    records = []
    
    phys_dir = CASE_DIR / 'data' / 'interpolated' / 'physiological'
    anno_dir = CASE_DIR / 'data' / 'interpolated' / 'annotations'
    
    if not phys_dir.exists():
        print(f"Warning: {phys_dir} not found.")
        return records

    phys_files = sorted(list(phys_dir.glob("sub_*.csv")))
    for p_path in phys_files:
        subj_id = p_path.stem
        a_path = anno_dir / f"{subj_id}.csv"
        try:
            if not a_path.exists():
                raise FileNotFoundError("Missing annotation file.")
                
            # Just read a few columns to check size
            df_p = pd.read_csv(p_path, usecols=['daqtime', 'ecg', 'gsr', 'video'])
            df_a = pd.read_csv(a_path, usecols=['jstime', 'valence', 'arousal'])
            
            # 10: baseline, 1/2: scary (stress), 3/4: amusing, 7/8: relaxed
            video_ids = df_p['video'].values
            n_raw = len(df_p)
            n_baseline = np.sum(video_ids == 10)
            n_active = np.sum((video_ids == 1) | (video_ids == 2)) # Negative Valence, High Arousal
            
            records.append({
                "dataset": "CASE",
                "subject_id": subj_id,
                "n_raw_samples_1000hz": n_raw,
                "n_baseline_samples": n_baseline,
                "n_active_samples": n_active,
                "baseline_duration_sec": n_baseline / 1000.0,
                "active_duration_sec": n_active / 1000.0,
                "available_channels": "ecg, gsr, video, valence, arousal",
                "status": "SUCCESS",
                "failure_reason": ""
            })
            write_access_log("CASE", subj_id, "SUCCESS")
        except Exception as e:
            records.append({
                "dataset": "CASE",
                "subject_id": subj_id,
                "status": "FAILED",
                "failure_reason": str(e)
            })
            write_access_log("CASE", subj_id, "FAILED")
            
    return records

def generate_integrity_report(df_audit):
    report = f"""# Dataset Integrity Report (Phase 2A)

- **Execution Date**: {datetime.datetime.now(datetime.timezone.utc).isoformat()}
- **WESAD Path Configured**: {WESAD_DIR.resolve()}
- **CASE Path Configured**: {CASE_DIR.resolve()}
- **Subjects Audited**: {len(df_audit)} total.

## WESAD Summary
- Audited Subjects: {len(df_audit[df_audit['dataset'] == 'WESAD'])}
- Baseline Definition: Class label 1 (Resting)
- Active Definition: Class label 2 (TSST Stress)
- Allowed Channels: ECG (chest), EDA (wrist)

## CASE Summary
- Audited Subjects: {len(df_audit[df_audit['dataset'] == 'CASE'])}
- Baseline Definition: Video ID 10 (bluVid)
- Active Definition: Video ID 1 & 2 (scary-1, scary-2 / High Arousal + Negative Valence)
- Allowed Channels: ecg, gsr

## Phase 2A Constraint Verification
- Classified Invoked: NO
- Normalization Applied: NO
- Performance Metric Computed: NO
"""
    with open("dataset_integrity_report.md", "w") as f:
        f.write(report)
    print("Generated dataset_integrity_report.md")

if __name__ == "__main__":
    if os.path.exists("real_data_access_log.jsonl"):
        os.remove("real_data_access_log.jsonl")
        
    records = audit_wesad() + audit_case()
    
    if records:
        df_audit = pd.DataFrame(records)
        df_audit.to_csv("real_data_ingestion_audit.csv", index=False)
        print("Generated real_data_ingestion_audit.csv")
        generate_integrity_report(df_audit)
    else:
        print("No records found. Check dataset paths.")
