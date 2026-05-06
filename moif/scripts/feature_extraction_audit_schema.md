# Feature Extraction Audit Schema

## Context
Leakage can occur before normalization ever runs. If operations like resampling, CWT, or cvxEDA decomposition utilize future samples or aggregate the entire timeseries prior to train/test splitting, the downstream validation is fatally compromised. This schema logs the strict causal bounds of feature extraction.

## Required JSON Schema (`feature_metadata`)

```json
{
  "feature_name": "string (e.g., 'eda_phasic_cvxEDA', 'bvp_hrv_hf')",
  "dataset": "string",
  "subject_id": "string",
  "feature_extraction_scope": "string (e.g., 'train_block_only', 'causal_rolling_window')",
  
  "leakage_audit": {
    "uses_full_timeseries_pre_split": "boolean",
    "uses_future_samples": "boolean",
    "uses_test_segment": "boolean",
    "uses_active_test_segment": "boolean",
    "causal_or_offline": "string (e.g., 'causal', 'offline_block')"
  },

  "processing_details": {
    "window_length_seconds": "float",
    "edge_handling": "string (e.g., 'pad', 'trim', 'mirror')",
    "resampling_method": "string (e.g., 'linear', 'nearest')",
    "smoothing_method": "string (e.g., 'moving_average_causal', 'butterworth_noncausal')"
  }
}
```

## Audit Verification Rules
Before the extracted features can be passed to the normalizer pipeline, the system must assert:
1. `leakage_audit.uses_full_timeseries_pre_split` MUST BE `false`.
2. `leakage_audit.uses_test_segment` MUST BE `false` (Unless the feature extractor is strictly refit independently on the test segment).
3. If a feature claims to be online/deployable, `leakage_audit.uses_future_samples` MUST BE `false`.
