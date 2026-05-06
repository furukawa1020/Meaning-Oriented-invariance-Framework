# Test Fixture Design (Adversarial Testing)

## Context
Unit tests are insufficient. We must build adversarial test fixtures to intentionally try to break our normalization bounds and induce leakage. If the normalizers survive these tests, the pipeline is secure.

## Fixture 1: The "+100" Active Injection
- **Design**: Create a dummy physiological timeseries. Set the `baseline` segment to a mean of `0`. Set the `active` segment to a mean of `+100`.
- **Target**: `baseline_train_only_zscore` and `baseline_covariance_whitening`.
- **Assertion**: After fitting and transforming the entire timeseries, the mean of the transformed `baseline` must be `0`. The mean of the transformed `active` segment MUST be approximately `+100`. If the active segment's transformed mean shifts towards `0`, it means the normalizer illegally accessed the active data during fitting.

## Fixture 2: The Future Peeking Test
- **Design**: Create a timeseries where a massive spike occurs at index `t = 1000`.
- **Target**: `rolling_zscore_60s`.
- **Assertion**: For all indices $t < 1000$, the transformed values must be completely identical regardless of whether the spike at $t=1000$ exists or is removed. If any value at $t=999$ changes, the rolling window is non-causal (peeking into the future).

## Fixture 3: The Singular Covariance Test
- **Design**: Create a multidimensional timeseries where two features are perfectly identical (correlation = 1.0), or one feature is entirely constant (variance = 0).
- **Target**: `baseline_covariance_whitening`.
- **Assertion**: The function must not crash with a `LinAlgError` (Singular Matrix). It must successfully apply the $\Sigma + \lambda I$ regularization, log the regularization event in the `fit_metadata`, and return a finite transformed array.

## Fixture 4: The Population Bleed Test
- **Design**: Create 3 subjects (A, B, C). Subject A has a mean of 100. B and C have means of 0.
- **Target**: `population_level_zscore`.
- **Assertion**: When transforming Subject A (held-out test subject), the scaler must be fit ONLY on B and C (mean = 0). Thus, Subject A's transformed data should preserve its relative offset (+100). If the scaler fits on A, B, C together, it will improperly center A's data.
