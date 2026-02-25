# Data directory contract (v0.1)

## WESAD placement

Place the official WESAD dataset here:

data/wesad/
  S2/
  S3/
  ...
  S17/

MOIF expects the original folder structure.

DO NOT commit raw WESAD data.
This directory is gitignored.

## Required structure (WESAD)

Each subject directory must contain:
- S*_Quest.csv
- S*_E4_Data/
- S*_Chest/

See WESAD original documentation.

## Data hash logging

At runtime, MOIF will compute SHA256 hashes of:
- Each subject folder (recursive)
- Or each CSV file (if CSV mode)

These hashes are stored in:
runs/{run_id}/data_hash.json

This guarantees reproducibility without storing raw data.
