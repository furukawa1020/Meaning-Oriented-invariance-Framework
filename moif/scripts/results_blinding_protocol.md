# Results Blinding Protocol

## Context
To guarantee the scientific integrity of the THMS submission, the analysis logic and Go/No-Go thresholds must be mathematically locked *before* the real WESAD and CASE datasets are processed. This prevents "p-hacking" or post-hoc threshold adjustment.

## The Freezing Mechanism
Before the script executes `moif_hms_eval.py` on real data, a locking script (`freeze_contract.py`) will run. This script will:
1. Hash the contents of `analysis_config.yaml`.
2. Hash the contents of `07_submission_go_no_go.md`.
3. Hash the contents of `normalization_audit_schema.md` and `feature_extraction_audit_schema.md`.
4. Output the hashes into a read-only `analysis_lock.md` file along with a precise timestamp.

## Allowed vs Forbidden Changes After Lock

### Allowed Changes (Post-Lock)
- Fixing computational errors (e.g., OutOfMemory crashes, path errors).
- Correcting code bugs that cause the adversarial tests to fail.
- Fixing typos in documentation.
- Altering plot aesthetics (colors, font sizes) without changing the underlying data.

### Forbidden Changes (Post-Lock)
- Lowering the submission Go/No-Go thresholds.
- Adding new metrics designed to "rescue" a failed hypothesis.
- Excluding specific "bad" subjects without pre-defined statistical justification.
- Swapping out the primary classifier (Logistic Regression) for a more complex one to brute-force better F1 scores.
- Altering the primary claim to fit an unexpected result.
