# Phase 2C: Pre-Run Contract

## Objective
This contract locks the operational workflow for Phase 2C (Main Performance Run) before any human sees empirical results from the real WESAD and CASE datasets.

## Mandatory Execution Order
1. Execute the Fixed Logistic Regression evaluation pipeline across all datasets.
2. The pipeline silently outputs the Phase 2C Result CSVs.
3. The evaluation script MUST IMMEDIATELY trigger `run_submission_gate.py`.
4. `run_submission_gate.py` evaluates the CSVs against the pre-registered thresholds and outputs `gate_report.json`.
5. **Human review is strictly blocked until `gate_report.json` is generated.**
6. No figures will be generated and no manuscript text will be written before the automated decision is logged.

## Phase 2C Output CSVs (Inputs to Gate)
- `results_main.csv` (Primary AUROC, AUPRC, F1, MCC per normalizer)
- `pipeline_disagreement.csv` (Cohen's Kappa, Label Flip Rates per normalizer pair)
- `results_subject_level.csv` (Subject-wise AUROC per normalizer)
- `calibration_length_results.csv` (AUROC per baseline calibration duration)
- `cross_dataset_transfer.csv` (Out-of-distribution robustness)

## Gate Logic & Thresholds (Mapped to `07_submission_go_no_go.md`)

### Gate 1: Normalization Effect
- *Pass Condition*:
  - (a) max-min AUROC or AUPRC across normalizers $\ge 0.03$, **AND**
  - (b) The effect is supported by at least 20% of subjects showing $|\Delta$AUROC$| \ge 0.05$ across normalizers, **OR** (a) holds across at least one dataset independently.
- *Required output*: `effect_source` field with one of `multiple_methods`, `baseline_vs_population`, `rolling_only`, or `unclear`.
  - If `effect_source == rolling_only`, the thesis claim must be weakened in the manuscript.

### Gate 2: Pipeline Disagreement
- *Pass Condition*: Mean prediction disagreement rate $\ge 0.05$, **OR** minimum Cohen's Kappa across pairs $\le 0.90$.

### Gate 3: Subject Heterogeneity
- *Pass Condition*: At least 20% of subjects exhibit $|\Delta$AUROC$| \ge 0.05$ across normalizers.

### Gate 4: Deployment Feasibility
- *Pass Condition* (either (a) or (b)):
  - (a) AUROC range across calibration lengths $\ge 0.03$, **OR**
  - (b) At least 20% of subjects show $|\Delta$AUROC$| \ge 0.05$ between the shortest and longest calibration length condition.

## Final Verdict Rules (3-Tier)
- **THMS_CANDIDATE**: All four gates pass. The results provide sufficient empirical support to proceed with a THMS manuscript draft.
- **NEEDS_MANUAL_REVIEW**: 2–3 gates pass, or results are mixed but potentially meaningful. Requires explicit human review of which gates failed and why before proceeding.
- **THMS_NOT_READY**: Fewer than 2 gates pass, or the primary thesis (normalization alters user-state modeling) is not supported by the data. Must route to a smaller venue or fundamentally redesign the study.
