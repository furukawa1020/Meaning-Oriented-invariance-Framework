# Phase 2B: Feature Extraction Gate Report

## Overview
This report evaluates the feature extraction audit outputs to determine if it is safe to proceed to Phase 2C (Main Performance Run). The core objective is to verify that the extraction logic (CWT, cvxEDA) does not inadvertently cause test-set data to leak into the training parameters.

## Audit Findings

### 1. Extraction Scope and Leakage (from `feature_extraction_audit_log.jsonl`)
- **Extraction Strategy**: Option A (Split-wise extraction) is mandated.
- **`uses_full_timeseries`**: `False`. Features are extracted strictly block-by-block.
- **`uses_future_samples`**: `True` (Locally). CWT/filtfilt uses forward-backward passes, meaning within a *single block*, future samples are used.
- **`uses_test_segment`**: `False`. Because the execution is split-wise, the forward-backward pass *never* crosses the boundary from the test segment into the train segment.

### 2. Edge Handling (from `feature_extraction_boundary_report.csv`)
- **`uses_samples_after_boundary`**: `False`.
- **`margin_discarded_sec`**: 5.0 seconds. To compensate for the edge artifacts caused by split-wise CWT/cvxEDA, a 5-second margin at the start and end of each extracted block is discarded before normalization.
- **Boundary Safety**: Confirmed `True`.

### 3. Feature Quality (from `feature_quality_report.csv`)
- **Missingness & Outliers**: Initial audit indicates 0% NaN/Inf creation under normal conditions, and outliers remain within the expected 1% physiological noise margin.
- **Constant Features**: 0 constant features detected. Covariance matrix invertibility (Mahalanobis) will be stable.

## Required Paper Revisions
Because we enforce Option A (split-wise extraction) instead of full-series extraction:
- **Constraint**: We must explicitly state in the THMS manuscript that "No cross-boundary test-set use was detected under the current split-wise extraction audit. The evaluation should be interpreted as split-wise offline blocked evaluation, not as a fully causal streaming pipeline."

## Go/No-Go Checklist Validation
- `[x]` CWTがtrain/test境界をまたいで未来情報を使わない (Blocked via Option A)
- `[x]` cvxEDAが全時系列一括処理ではない (Blocked via Option A)
- `[x]` feature extraction scope がsubjectごとに一致 (All subjects use split-wise Option A)
- `[x]` NaN/Infが多くない (0% critical missingness)
- `[x]` 定数特徴がなく、covariance calibrationが安定 (Confirmed)
- `[x]` WESAD/CASEで共通特徴集合が定義できる (EDA Phasic, HRV LF/HF)
- `[x]` 欠損処理が事事前ルールに基づく (Edge handling is strictly defined at 5s)

## Conclusion
Current audit parameters confirm that no test-segment data crossed into the training parameters during extraction. 
**Conditional Go granted for Phase 2C-0 Gate Dry Run.**
