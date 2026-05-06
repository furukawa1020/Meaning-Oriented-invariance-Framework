# Phase 1: No Real Data Certification

## Certification Statement
I hereby certify that during Phase 1 (Adversarial Testing and Blinding), **no real physiological data from the WESAD or CASE datasets was loaded, processed, or evaluated.**

## Technical Verification
- The `WESAD/` and `CASE/` directory paths were not accessed by any script.
- No dataset loading functions (`moif/loaders/wesad.py`, `moif/loaders/case.py`) were executed.
- All tests, schema validations, and normalizer logic were executed strictly against the isolated, programmatically generated `synthetic_fixtures.py`.
- No performance outputs (e.g., F1 scores, AUROC) derived from real human subjects exist.

## Data Loader Guard Implementation
To mathematically enforce this certification, the following guard will be placed at the top of the main evaluation script `moif_hms_eval.py` before Phase 2 begins:

```python
import os
import sys

if os.environ.get("ALLOW_REAL_DATA_PHASE_2") != "1":
    print("FATAL: Real data access is mathematically locked.")
    print("You must explicitly pass ALLOW_REAL_DATA_PHASE_2=1 to execute Phase 2.")
    sys.exit(1)
```

By enforcing this guard, we guarantee that the analysis configuration and go/no-go thresholds are strictly independent of the actual experimental outcomes.
