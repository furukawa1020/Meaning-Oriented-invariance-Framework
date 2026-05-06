# 04 Main Results Interpretation Framework

## Context within THMS Paper
This document outlines how we will interpret the raw CSV outputs once Phase 1-5 is executed. We strictly avoid claiming "Method X is superior." Instead, we evaluate results through the lens of the Submission Gates.

## Gate 1: Performance & Subject-Level Variability
- **Expected Outcome**: Rolling Z-score shows significant F1/AUROC degradation compared to static methods. Covariance calibration and Baseline-only Z-score match or approximate Subject-wise Z-score performance.
- **Interpretation**: If rolling fails, it proves that "dynamic normalization can suppress sustained offsets crucial for state modeling." If baseline methods succeed, it proves that "active-state statistics are not strictly necessary if resting baseline variance is properly modeled."
- **Subject Variability**: If standard deviations are high and paired plots show specific subjects failing under certain normalizations, it proves that "normalization effects are highly personalized, complicating population-level HMS deployment."

## Gate 2: Pipeline Disagreement
- **Expected Outcome**: Cohen's Kappa is substantially lower than 1.0 between different normalization pipelines, and risk-score correlations break down.
- **Interpretation**: This is the core thesis. "Even holding the physiological signal, the classifier, and the subjective label constant, altering the preprocessing pipeline changes the machine's interpretation of the user state." The physiological signal is therefore NOT a stable label; it is highly context-dependent.

## Gate 3: Feature-Space Geometry
- **Expected Outcome**: Mahalanobis distances between Baseline/Active classes collapse under rolling windows but remain separated under static standardizations.
- **Interpretation**: "Normalization physically distorts the state separability geometry before the classifier even sees the data."

## Gate 4: Cross-Dataset Invariance Stress Test
- **Expected Outcome**: Models trained on WESAD fail to generalize effectively to CASE, and feature contribution rankings (LR coefficients) diverge.
- **Interpretation**: "Physiological responses map to specific task/context formulations (acute social stress vs video-induced arousal). The lack of cross-dataset invariance further supports that physiological signals are context-dependent evidence, not universal biological markers of subjective states."
