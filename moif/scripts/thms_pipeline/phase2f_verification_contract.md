# Phase 2F: Model-Capacity Effect Verification Contract

## Objective
To verify if the exploratory "Random Forest boost" observed in Phase 2E is a robust phenomenon across all 45 subjects and multiple evaluation units (window-level), or if it was an artifact of sample-level autocorrelation and ceiling effects.

## Terminology Standard
- **PROHIBITED**: "DBA", "Proposed method", "DBA論文".
- **MANDATORY**: "resting-baseline covariance calibration" or "baseline-conditioned covariance calibration".

## Evaluation Protocol
### 1. Primary Comparison
- **Normalizers**:
    - `covariance_calibration` (Baseline-conditioned covariance whitening)
    - `baseline_z` (Z-score fitted on baseline-only)
    - `subject_z` (Z-score fitted on all training data)
- **Models**:
    - **Primary**: Random Forest (RF) - n_estimators=50, depth=10.
    - **Reference**: Logistic Regression (LR), Linear SVM.

### 2. Datasets
- **WESAD**: All 15 subjects.
- **CASE**: All 30 subjects (treated as auxiliary/validation).

### 3. Evaluation Units (CRITICAL)
- **Sample-level (100Hz)**: For reference only.
- **Window-level (Primary)**: 1-second non-overlapping windows (mode of 100 samples) or mean feature vector per window.
- **Ceiling Exclusion**: Any subject-feature-model combination yielding AUROC >= 0.98 is flagged as "ceiling" and excluded from the primary delta aggregation to avoid misleadingly perfect results.

### 4. Metrics
- AUROC, AUPRC, Balanced Accuracy, MCC.

## Success Gates (Go/No-Go for BSPC Manuscript)
- **Gate F1: Within-Dataset Effect**: Mean $\Delta$AUROC $\ge 0.03$ compared to Z-score baselines in either WESAD or CASE, with 95% CI strictly $> 0$.
- **Gate F2: Subject Support**: At least 30% of subjects must exhibit individual $\Delta$AUROC $\ge 0.03$.
- **Gate F3: Window-Level Robustness**: The improvement must persist at the 1-second window evaluation level.
- **Gate F4: Model Specificity**: If the effect is only present in RF, it must be reported as "model-capacity-dependent."
- **Gate F5: Implementation Integrity**: No leakage, same splits, same samples across all normalizers.

## Decision Mapping
- **PASS F1-F3**: Proceed to BSPC manuscript drafting focusing on "Model-Capacity-Dependent Effects."
- **FAIL F1-F3**: Terminate the pursuit. Document as "Exploratory signal was an artifact/unreliable." THMS is definitively out of scope.
