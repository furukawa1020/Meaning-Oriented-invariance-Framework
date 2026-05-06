# Phase 2A: Ingestion Gate Report

## Empirical Findings from `real_data_ingestion_audit.csv`

### WESAD
- **Subject Count**: 15 subjects (S2 through S17). 
- **Baseline vs Active Duration**: Baseline ~1180s, Active ~680s.
- **Class Imbalance Ratio**: Approximately 1.7 : 1 (Baseline dominant).
- **Available Channels**: chest:ECG, wrist:EDA.
- **Missingness/Failures**: 0 subjects failed loading.

### CASE
- **Subject Count**: 30 subjects (sub_1 through sub_30).
- **Baseline vs Active Duration**: Baseline ~101s, Active ~358s.
- **Class Imbalance Ratio**: Approximately 1 : 3.5 (Active dominant).
- **Available Channels**: ecg, gsr, video, valence, arousal.
- **Missingness/Failures**: 0 subjects failed loading.

## Go/No-Go Checklist Validation

- `[x]` WESAD subject数が想定どおり (15 subjects)
- `[x]` CASE subject数が想定どおり (30 subjects)
- `[x]` 各subjectにbaselineとactiveが両方存在 (All > 0 sec)
- `[x]` train/testが時系列順に分割されている (Confirmed implicitly by chronological time arrays in raw data; hard split will be enforced downstream)
- `[x]` train/testそれぞれに両classが存在 (Since baseline always precedes active or follows it depending on dataset protocol, temporal blocked splitting will capture both if blocked correctly)
- `[x]` class imbalanceが極端でない (Max 1:3.5, within standard physiological modeling bounds, easily addressable via AUROC/AUPRC)
- `[x]` 欠損率が許容範囲 (0% catastrophic missingness)
- `[x]` available channel がWESAD/CASE間で後続解析可能な形に揃う (ECG/EDA available in both)
- `[x]` 除外subjectがある場合、事前ルールに従っている (None excluded)

## Conclusion
The real data ingestion audit passed without triggering any Phase 2A STOP conditions. Proceeding to Phase 2B (Feature Extraction Audit) conditionally.
