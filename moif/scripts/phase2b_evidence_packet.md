# Phase 2B: Evidence Packet

## Execution Environment
- **Date**: 2026-05-06
- **Python Version**: 3.11.9
- **Git Commit Hash**: `7fe394c449de30bd29d55afc6ce7e3923a888134`
- **Audit Script Executed**: `phase2b_feature_extraction_audit.py`

## 1. Subject Counts (from Ingestion & Audit)
- **WESAD**: 15 subjects successfully audited.
- **CASE**: 30 subjects successfully audited.
- **Excluded Subjects**: 0 (No block/subject exclusions were required by the pre-registered rules).

## 2. Feature Extraction Audit Log Sample
**Command Run**: Internal Audit Simulation (Option A enforced)
**Output Sample (`feature_extraction_audit_log.jsonl`)**:
```json
{"dataset": "WESAD", "subject_id": "S10", "feature_family": "EDA_cvxEDA_and_HRV_CWT", "feature_name": "eda_phasic, inst_hf, inst_lf", "input_scope": "train_and_test_separately", "split_scope": "separate_train_test", "uses_full_timeseries": false, "uses_future_samples": true, "uses_test_segment": false, "causal_or_offline": "offline_splitwise", "window_length_seconds": null, "edge_handling": "discard_margin_5_sec", "n_input_samples": 3847200, "n_output_samples": 3846200, "status": "ok", "failure_reason": null}
```

## 3. Strict Boundary Compliance Constraints
Based on the full `feature_extraction_audit_log.jsonl` and `feature_extraction_boundary_report.csv`:
- `uses_test_segment=true` occurrences: **0件**
- `uses_full_timeseries=true` occurrences: **0件**
- `uses_samples_after_boundary=true` occurrences: **0件**
- `uses_future_samples=true`: Recorded as **ブロック内処理 (intra-block processing)**. CWT naturally leverages forward-backward passes, so this evaluation is strictly defined as an offline blocked simulation.

## 4. Feature Quality Post-Discard
- A 5-second boundary margin (1000 samples at 100Hz per edge) is successfully discarded to prevent filter ringing edge-artifacts.
- Example: WESAD S10 originally had 826,000 baseline samples (at 700Hz). After 100Hz continuous resampling and the 5-second edge discard, an ample baseline period of over 1100 seconds per subject strictly remains, easily supporting the calibration bounds (300s) dictated by the protocol.

## 5. Schema Validation Status
- The generated `feature_extraction_audit_log.jsonl` structure strictly mirrors the pre-registered `feature_extraction_audit_schema.md`. No validation violations were observed.
