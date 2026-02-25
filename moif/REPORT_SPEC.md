# REPORT_SPEC — MOIF v0.1 レポート仕様

Report is a reproducibility artifact. It must include:

## 1. Run metadata
- run_id
- git commit hash
- python version
- dependency freeze
- config summary
- input data hashes

## 2. Detection summary table
For each finding:
- signal
- band definition (abs/norm + params)
- condition contrast (c1 vs c2)
- divergence (metric + value)
- p_value, q_value, effect_z
- sample counts per condition

## 3. Top condition explanations
- ranked features with scores

## 4. Generated protocol links
- protocol.yaml
- run_sheet.md
- analysis_stub.ipynb
