# Phase 2D: Anomaly Audit & Decision Report

## Audit Results Summary

### 1. Gate 1: Normalization Effect (Decomposition)
- **Global Range (AUROC)**: 0.1857
- **Within-WESAD Range**: 0.0039 (Fail)
- **Within-CASE Range**: 0.0058 (Fail)
- **Effect Source**: **`dataset_offset`**
- **Conclusion**: The primary effect observed in the gate report was an artifact of the performance difference between WESAD and CASE datasets. Within each dataset, the choice of normalization had negligible impact on aggregate AUROC/AUPRC.

### 2. Gate 2: Pipeline Disagreement (Audit)
- **Mean Disagreement Rate**: 1.3% (Very Low)
- **Minimum Cohen's Kappa**: 0.160
- **Conclusion**: The low Kappa is likely a result of high class imbalance and the "Kappa Paradox," rather than meaningful interpretive instability. With only 1.3% of labels flipping, the pipeline-level impact is insufficient for a strong THMS claim.

### 3. Gate 4: Deployment Feasibility (Anomaly)
- **Min AUROC**: 0.0 (WESAD_S2_120s)
- **Max AUROC**: 1.0 (WESAD_S10_30s)
- **Conclusion**: **Confirmed Anomaly.** AUROC values of 0.0 and 1.0 at short calibration lengths indicate numerical instability, likely due to insufficient samples or single-class presence in the test segments at those specific durations. This gate's "Pass" was a false positive driven by instability.

### 4. Gate 3: Subject Heterogeneity (Failure)
- **Percent >= 0.05**: 6.67% (**Hard Fail**)
- **Percent >= 0.03**: 17.78%
- **Median Delta AUROC**: 0.0000
- **Conclusion**: The impact of normalization at the individual subject level is nearly non-existent for the vast majority of participants.

---

## Final Decision: THMS Withdrawal

Based on the Phase 2D audit, the empirical evidence **does not support** the core thesis ("Normalization is part of the user-state model") required for a 10-page THMS Regular Paper. 

- **Status**: **THMS SUBMISSION HALTED.**
- **Reason**: The observed effects are either dataset-level offsets or numerical anomalies. No robust "Human-Machine System" interaction (subjective variability or interpretive instability) was detected.

## Recommended Next Steps
1. **Redesign/Pivot**: Investigate if the feature extraction (Phase 2B) or the fixed-parameter Logistic Regression (Phase 2C) is overly regularized, masking potential effects.
2. **Alternative Venue**: If the results remain stable after verification, pivot to a signal-processing focused journal (e.g., *Biomedical Signal Processing and Control*) with a significantly narrowed scope focusing on dataset-level benchmarking rather than user-state modeling.
3. **Debug**: Resolve the AUROC 0.0/1.0 instability in the calibration evaluation code before any further runs.
