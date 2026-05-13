## Overview

This archive contains the supplementary materials for the manuscript:

**"LabelScope: A Claim-Capping Framework for Auditing Operational Labels in Physiology-Based Human-Machine Systems"**

Submitted to: IEEE Transactions on Human-Machine Systems

> **Note on file contents**: The CSV files in `results/` contain audit result summaries and per-subject performance metrics as reported in the manuscript tables. Raw physiological signal data and window-level prediction records are not included in this archive due to dataset licensing restrictions (WESAD and SWELL-KW are publicly available; see `data_instructions/` for download and preparation steps). The Python scripts in `scripts/` implement the audit logic described in the manuscript and are intended to be run after dataset preparation.

## Directory Structure

```
supplementary_archive/
├── README.md                         (this file)
├── configs/
│   ├── preprocessing_config.json     (signal processing parameters)
│   ├── audit_config.json             (audit module thresholds)
│   └── random_seeds.json             (reproducibility seeds)
├── scripts/
│   ├── bootstrap_script.py           (subject-cluster bootstrap CI)
│   ├── sa_implementation.py          (Structure Audit core implementation)
│   ├── sa_negative_control.py        (SA negative control procedures)
│   ├── candidate_restricted_sa.py    (temporal-neighbor-excluded SA)
│   ├── swell_aggregation.py          (SWELL-KW block-level aggregation)
│   └── table_generation.py           (automated results table generation)
├── results/
│   ├── loso_predictions_wesad.csv    (LOSO per-subject predictions)
│   ├── bootstrap_confidence_intervals.csv
│   ├── wesad_pa_results.csv          (WESAD Proxy Audit outputs)
│   ├── wesad_sa_results.csv          (WESAD Structure Audit outputs)
│   ├── candidate_restricted_sa_results.csv
│   ├── ra_pa_threshold_sensitivity.csv
│   ├── swell_kw_pa_results.csv
│   └── swell_kw_performance_results.csv
├── data_instructions/
│   ├── WESAD_download_and_preparation.md
│   └── SWELL_KW_download_and_preparation.md
└── figures/
    ├── fig4_dashboard.pdf
    └── fig5_sensitivity.pdf
```

## Reproducibility

All random seeds are fixed and documented in `configs/random_seeds.json`.
To reproduce results, install dependencies and run scripts in the order:
1. Preprocess signals using parameters in `configs/preprocessing_config.json`
2. Run `scripts/bootstrap_script.py` for CI estimation
3. Run `scripts/sa_implementation.py` for SA module
4. Run `scripts/table_generation.py` to reproduce paper tables

## Datasets

WESAD and SWELL-KW datasets are publicly available. See `data_instructions/` for download links and preparation steps. Raw data files are not included in this archive due to licensing restrictions.

## Contact

For questions, please contact the corresponding author.
