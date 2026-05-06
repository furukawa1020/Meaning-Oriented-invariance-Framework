# Normalization Audit Schema

## Context
To mathematically guarantee that no active-state or future test-state data leaked into our normalization scalers, every normalization transform function must return both the `transformed_X` array and a strict `fit_metadata` JSON audit log.

## Required JSON Schema (`fit_metadata`)

```json
{
  "normalizer_name": "string (e.g., 'baseline_train_only_zscore')",
  "fit_subjects": ["list of subject IDs used to fit"],
  "fit_time_range_seconds": [0.0, 1200.0],
  "fit_labels_used": ["list of unique labels present in the fit data", e.g., ["baseline"]],
  "fit_split": "string (e.g., 'train')",
  "n_samples_fitted": 120000,
  "feature_dim": 8,
  
  "leakage_audit": {
    "active_train_used": "boolean",
    "active_test_used": "boolean",
    "baseline_train_used": "boolean",
    "baseline_test_used": "boolean",
    "other_subjects_used": "boolean",
    "future_samples_used": "boolean"
  },

  "transform_audit": {
    "contains_nan": "boolean",
    "contains_inf": "boolean",
    "covariance_condition_number": "float (null if not covariance-based)"
  }
}
```

## Audit Verification Rules
Before any performance metrics are calculated, the pipeline must assert:
1. If `normalizer_name` == `baseline_train_only_zscore` or `baseline_covariance_whitening`:
   - `leakage_audit.active_train_used` MUST BE `false`.
   - `fit_labels_used` MUST NOT contain `active`.
2. For ALL normalizers (except Population-level):
   - `leakage_audit.other_subjects_used` MUST BE `false`.
3. For ALL normalizers:
   - `leakage_audit.active_test_used` MUST BE `false`.
   - `leakage_audit.baseline_test_used` MUST BE `false`.
   - `leakage_audit.future_samples_used` MUST BE `false` (for rolling causal).
