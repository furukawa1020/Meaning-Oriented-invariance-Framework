# 03 Leakage Control Protocol

## Context within THMS Paper
A critical failure of many physiological sensing papers is temporal leakage (normalizing the entire timeseries globally before splitting into train/test). This section details our strict `leakage-controlled temporally blocked evaluation` to guarantee the validity of the HMS implications.

## Evaluation Design

### Temporal Blocked Split (50/50)
For each subject, the continuous physiological timeseries is split chronologically:
- **Training Block**: The first 50% of the timeline (containing the initial baseline and the first half of the active state).
- **Testing Block**: The remaining 50% of the timeline (containing the second half of the active state and recovery).

### Parameter Estimation Isolation
- **Rule 1**: Normalization parameters ($\mu, \sigma, \Sigma$) are estimated **strictly** from the designated segments within the Training Block.
- **Rule 2**: For `Baseline-Only` and `Covariance Calibration`, the parameters are estimated *exclusively* from the baseline portion of the Training Block. The active portion of the Training Block is strictly ignored during normalization fitting.
- **Rule 3**: The Testing Block is completely unseen during normalization fitting and classifier training.

## Required Unit Tests
Before generating results, the python pipeline MUST pass the following automated assertions:
- `test_baseline_only_uses_no_active_statistics`
- `test_subject_trainblock_uses_training_only`
- `test_no_test_statistics_used_in_normalization`
- `test_rolling_zscore_is_causal` (no future lookahead)

## Manuscript Figure
We will present a conceptual diagram (Fig. 2) showing the timeline, the 50/50 split, and precisely which segments are used to estimate the normalizers for each method.
