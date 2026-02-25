# MOIF v0.1 (Meaning-Oriented Invariance Framework)

MOIF is an OSS research engine to **detect, reproduce, and verify invariance breaking**
in the mapping from physiological signals (X) to subjective labels (Y), conditioned by context (C).

## Install (dev)
```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

## Quickstart

```bash
moif detect --config configs/wesad_hr_absband.yaml --out runs/demo
moif report --run runs/demo/latest
moif sced --run runs/demo/latest --type abab
```

## Output contract

* `runs/{run_id}/config.yaml` (fully resolved)
* `runs/{run_id}/env.json` (python/pip/git)
* `runs/{run_id}/findings.jsonl` (Finding schema)
* `runs/{run_id}/run_result.json` (RunResult schema)
* `runs/{run_id}/report.md` (rendered)

See `schemas/` and `REPORT_SPEC.md`.

## Scope (v0.1)

* WESAD loader + CSV loader
* Band-based invariance breaking detection (divergence + permutation + FDR)
* Condition explanation (ranking only)
* SCED protocol template generation
* Reproducibility-first logging
