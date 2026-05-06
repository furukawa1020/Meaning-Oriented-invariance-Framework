# 05 Human-Machine Systems (HMS) Implications

## Context within THMS Paper
This is the most critical section for THMS. It translates the signal processing findings into systemic engineering, validity, and interpretability consequences for the HMS community.

## Core Discussion Points

### 1. Validity of User-State Inference
Physiological signals are widely used to drive adaptive UIs, workload monitors, and stress interventions. Our results demonstrate that the validity of these systems relies heavily on the chosen normalization strategy. Normalization is not a neutral mathematical operation; it imposes assumptions about baseline drift and physiological responsivity.

### 2. Interpretability and System Disagreement
If the exact same continuous data stream yields a "stressed" prediction under rolling normalization but a "baseline" prediction under baseline-only calibration, the interpretability and accountability of the HMS are compromised. System designers must recognize that the "machine's view" is pipeline-dependent.

### 3. Deployment-Constrained Calibration
Subject-wise standardization, while theoretically strong, is practically flawed for HMS deployment because it requires future knowledge of the user's active state. Our findings validate that `baseline-only` strategies are viable deployment alternatives that respect the causal flow of real-world sensing.

### 4. Reporting Recommendations for HMS Research
To ensure reproducibility and operational transparency in future HMS studies utilizing physiological sensors, we recommend the mandatory reporting of:
1. The exact normalization method.
2. The specific statistics (baseline vs active) used for estimation.
3. The temporal leakage-control protocol (whether the scaler "saw" future test data).
4. Causal deployability (online vs offline).
