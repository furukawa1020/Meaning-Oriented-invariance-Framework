# DEV_SPEC — MOIF v0.1 開発定義書

## 0. 不変の目的
生理信号Xと主観/ラベルYの対応が、条件Cによって変わる（不変性が破れる）ことを
**検出・再現・検証できる再現可能OSS基盤**として固定する。

v0.1は「推定精度」ではなく「検証可能性」を成果物とする。

---

## 1. コア概念の定義（v0.1で固定）
- X(t): physiological signal (HR, EDA)
- Y: label (stress/amusement/neutral)
- C: condition candidates (task, time_of_day, session_id, subject_id)
- Band B: "same physiological state" range definition

### Invariance hypothesis
P(Y | X∈B) is invariant to C.

### Invariance breaking
There exists c1,c2 such that:
P(Y | X∈B, C=c1) != P(Y | X∈B, C=c2)

---

## 2. v0.1 スコープと禁止事項
### Included
- WESAD loader + generic CSV loader
- Band extraction (abs or normalized)
- Divergence computation (fixed metric)
- Permutation test + FDR correction (BH)
- Condition explanation as ranking (not predictive claim)
- Externalization abstract output (device-agnostic)
- SCED protocol templates (ABAB, randomized switch)
- Reproducibility logging (config/seed/env/datahash)

### Excluded (禁止)
- device control / app UI / clinical claims
- multi-sensor fusion beyond WESAD (new sensors)
- "best model" competition
- unlogged preprocessing (every transform must be logged)

---

## 3. システムI/O契約（Schemasが真実）
All contracts must be represented as JSON Schema in `schemas/`.
Implementation must not silently change output formats.

### Required schemas
- config.schema.json
- finding.schema.json
- run_result.schema.json
- external_state.schema.json
- sced_protocol.schema.json

---

## 4. パイプライン仕様
Input -> Signal Processing -> Banding -> Invariance Detection -> Condition Explain
-> Externalization -> SCED Template -> Artifacts

### Artifact root
runs/{run_id}/

#### Must outputs (v0.1)
- config.yaml (resolved)
- env.json (python/pip/git)
- data_hash.json (hashes of input files)
- findings.jsonl (one JSON per finding)
- run_result.json (summary)
- report.md (rendered)

---

## 5. Banding（最重要：曖昧禁止）
v0.1 supports two modes, but each run must choose exactly one:
- abs: [low, high] in original unit (e.g., HR bpm)
- norm: z-score band or quantile band per subject

Config must specify:
- mode: abs|norm
- parameters: {low, high} or {z_low, z_high} or {q_low, q_high}
- min_n_per_condition: minimum samples per condition per band

---

## 6. Divergence & Significance（検出の契約）
### Divergence metric (fixed in v0.1)
- JSD (Jensen-Shannon divergence) on categorical Y distribution

### Significance
- permutation test (shuffle condition assignment)
- p-value computed from null distribution
- multiple testing correction: BH(FDR), produce q-value

Effect size in v0.1:
- effect_z = (obs - mean(null))/std(null)

---

## 7. Condition explanation（主張を限定）
Goal: provide **ranking of condition features** that best explain divergence.
Not to claim generalizable prediction.

Implementation (v0.1):
- decision tree or logistic regression
Output:
- feature importances / coefficients
- CV settings logged

---

## 8. Externalization abstraction（デバイス非依存契約）
ExternalState schema is fixed.
Mapping functions are plugins, but output ranges must be defined:
- arousal_level in [0,1]
- divergence_index in [0,1]
- confidence in [0,1]

No claim of "optimal mapping" in v0.1.

---

## 9. SCED template generation（生成物の仕様）
v0.1 generates runnable templates:
- protocol.yaml (phase plan, switch rules, measures)
- run_sheet.md (human-readable steps)
- analysis_stub.ipynb (analysis skeleton)

Support:
- ABAB
- randomized_switch (with seed-logged schedule)

---

## 10. テスト方針（v0.1雛形の守るべき線）
Tests must ensure:
- configs validate against config schema
- findings validate against finding schema
- run artifact folder created with required files
- report rendering produces deterministic structure

---

## 11. CI要件
GitHub Actions must run:
- ruff (lint)
- pytest

No heavy datasets in CI.
Use small synthetic fixtures in tests.

---

## 12. 成功判定（v0.1）
- CLI works: detect/report/sced
- Artifacts generated & schema-valid
- Third party can clone and run demo pipeline (with synthetic or small sample)
- WESAD notebook skeleton exists (full data run documented, not CI)

---

## 13. Data reproducibility (追加)

### 13.1 WESAD mode

When dataset.type == "wesad":

* Recursively scan `data/wesad/`
* Compute SHA256 for each:

  * Subject directory
  * CSV file inside subject
* Store results in:

```
runs/{run_id}/data_hash.json
```

### data_hash.json structure:

```json
{
  "dataset_type": "wesad",
  "root": "data/wesad",
  "files": {
    "S2/S2_Quest.csv": "sha256hash...",
    "S2/S2_Chest/HR.csv": "sha256hash..."
  }
}
```

### 目的

* Rawデータをコミットせず
* 再現性を保証
* 別マシンでも同一データであることを検証可能

---

## 14. WESAD Loader Contract

`moif/loaders/wesad.py` must:

* Accept root path: `data/wesad`
* Enumerate subjects dynamically (S2, S3, ...)
* Extract:

  * HR (Chest)
  * EDA (E4)
  * label per time segment
* Align timestamps explicitly
* Log alignment method in run_result.json

### Output unified dataframe columns:

| column      | type  |
| ----------- | ----- |
| timestamp   | float |
| subject_id  | str   |
| session_id  | str   |
| task        | str   |
| signal_name | str   |
| value       | float |
| label       | str   |

No silent interpolation allowed.
All preprocessing steps must be logged.
