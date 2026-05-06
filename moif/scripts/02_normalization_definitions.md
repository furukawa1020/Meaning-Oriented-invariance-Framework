# 02 Normalization Definitions

## Context within THMS Paper
Normalization is treated not as a mathematical necessity, but as a specific assumption about human physiology. This section defines the 5 main normalization strategies, explicitly categorizing them by the *information* they require and their *online deployability*.

## Strategy Definitions

### 1. Subject-Wise Train-Block Z-Score
- **Statistics Used**: Mean and standard deviation of the *entire* training block (which includes both baseline and active states).
- **Active-State Statistics Required?**: **Yes**.
- **Online Deployable?**: Low. It assumes prior exposure to the target active state before real-world inference begins.

### 2. Baseline-Only Z-Score
- **Statistics Used**: Mean and standard deviation of the *resting baseline* segment only.
- **Active-State Statistics Required?**: **No**.
- **Online Deployable?**: High. It only requires a short calibration period before the task begins.

### 3. Resting-Baseline Covariance Calibration (Formerly DBA)
- **Statistics Used**: Mean vector and Covariance matrix of the *resting baseline* segment only. Applies a multi-dimensional Mahalanobis whitening transformation.
- **Active-State Statistics Required?**: **No**.
- **Online Deployable?**: High. Requires only a short multidimensional calibration period.

### 4. Rolling Z-Score (30s / 60s / 120s)
- **Statistics Used**: Mean and standard deviation computed over a continuous causal past window (e.g., past 60 seconds).
- **Active-State Statistics Required?**: **No**.
- **Online Deployable?**: High, but mathematically unstable as it suppresses sustained physiological offsets (treating the sustained active state as the new baseline).

### 5. Population-Level Scaling (Training-Population Z-Score)
- **Statistics Used**: Mean and standard deviation estimated from all subjects in the training set, explicitly *excluding* the held-out test subject.
- **Active-State Statistics Required?**: **Yes** (from other subjects).
- **Online Deployable?**: High (can be pre-computed), but ignores intra-subject baseline variance entirely.

## Reporting Requirements
Table 2 in the manuscript will map these methods directly to their "Uses resting baseline?", "Uses active-state statistics?", "Causal?", and "Deployment realism" traits.
