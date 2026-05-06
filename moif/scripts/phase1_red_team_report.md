# Phase 1: Red Team Report on Adversarial Testing Framework

## 1. Executive Summary
This document critiques the adversarial testing framework implemented in Phase 1. The goal of the framework was to guarantee that no "leakage" (the use of active-state or future-state information in normalizer parameters) occurs. This report evaluates the validity of the tests themselves.

## 2. Verification of Bad Implementations
We purposefully authored `bad_normalizers.py` containing:
- Baseline calibrations that fail to filter out the `active` split.
- Rolling windows that use `center=True` (peeking into the future).
- Population scalers that fail to exclude the held-out test subject.

**Observation**: `run_bad_implementations_should_fail.py` confirmed that 4/4 of these intentionally leaky implementations were caught by the adversarial tests.
**Mechanism of Capture**: The tests inject massive perturbations (e.g., +1000) into the strictly forbidden segments (like the future, or the active train segment). If the normalizer is leaky, these perturbations pollute the parameters ($\mu, \sigma$), drastically altering the transformed output of the *unperturbed* segments.

## 3. Verification of Correct Implementations
The correct implementations (`normalizers.py`) passed the exact same test suite.
**Mechanism of Pass**: Because they strictly isolate the allowed training data (e.g., filtering strictly by `condition == 'baseline'`), the massive perturbation injected into the active train segment is completely ignored during parameter fitting. Thus, the baseline transformation remains perfectly identical to the unperturbed state.

## 4. Red Team Vulnerability Assessment (Blind Spots)
While the adversarial test suite is robust, it has the following theoretical blind spots:

### A. Synthetic Fixture Limitations
The synthetic fixture (`synthetic_fixtures.py`) generates pure Gaussian noise with fixed offsets.
- **Blind Spot**: Real physiological data has long-range temporal dependencies (autocorrelation). The tests do not check if a normalizer accidentally destroys or leverages low-frequency autocorrelation, because the synthetic data has none.

### B. Feature Extraction Pre-Leakage
- **Blind Spot**: The current tests only evaluate the *normalizers* (given a $N \times D$ feature matrix). If the preceding feature extraction step (e.g., continuous wavelet transform, or cvxEDA) was run over the entire timeseries *before* splitting, the features themselves are already contaminated.
- **Mitigation**: We introduced `feature_extraction_audit_schema.md` to mandate logging of feature scope, but we do not currently have adversarial tests for the CWT/cvxEDA code itself, as that relies on raw 100Hz data ingestion which is locked behind Phase 2A.

### C. Numerical Instability in Near-Singular Covariance
- **Blind Spot**: `test_09_singular_covariance` proves that regularization prevents a crash. However, it does not prove that the resulting spatial transformation is *physiologically meaningful*. If the baseline data has zero variance in one dimension, the regularized Mahalanobis distance might drastically inflate noise in that dimension during the active phase.

## 5. Conclusion
The adversarial tests provide a mathematical guarantee against temporal and condition-based leakage in the normalization stage. However, interpretation of the downstream results must remain cautious regarding feature-extraction leakage and the physiological validity of regularized spatial transformations.
