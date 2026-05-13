"""
bootstrap_script.py
Subject-cluster bootstrap for confidence interval estimation.

Usage:
    python bootstrap_script.py --input results/wesad_pa_results.csv \
                               --output results/bootstrap_confidence_intervals.csv \
                               --n_bootstrap 1000 --seed 42
"""
import numpy as np
import pandas as pd
import argparse
import json
from pathlib import Path


def subject_cluster_bootstrap(values, subject_ids, n_bootstrap=1000, seed=42):
    """
    Performs subject-cluster bootstrap resampling.

    Args:
        values: array-like of per-subject estimates (e.g., proxy indices)
        subject_ids: array-like of subject identifiers
        n_bootstrap: number of bootstrap iterations
        seed: random seed for reproducibility

    Returns:
        ci_lower: lower bound of 95% CI
        ci_upper: upper bound of 95% CI
        bootstrap_means: array of bootstrap mean estimates
    """
    rng = np.random.default_rng(seed)
    unique_subjects = np.unique(subject_ids)
    n_subjects = len(unique_subjects)
    bootstrap_means = np.zeros(n_bootstrap)

    for b in range(n_bootstrap):
        sampled_subjects = rng.choice(unique_subjects, size=n_subjects, replace=True)
        sampled_values = []
        for s in sampled_subjects:
            idx = np.where(subject_ids == s)[0]
            sampled_values.extend(values[idx].tolist())
        bootstrap_means[b] = np.mean(sampled_values)

    ci_lower = np.percentile(bootstrap_means, 2.5)
    ci_upper = np.percentile(bootstrap_means, 97.5)
    return ci_lower, ci_upper, bootstrap_means


def main():
    parser = argparse.ArgumentParser(description="Subject-cluster bootstrap CI estimation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_bootstrap", type=int, default=1000)
    args = parser.parse_args()

    # Load random seeds from config
    seeds_path = Path("configs/random_seeds.json")
    if seeds_path.exists():
        with open(seeds_path) as f:
            seeds = json.load(f)
        seed = seeds.get("bootstrap_seed", args.seed)
        n_bootstrap = seeds.get("n_bootstrap", args.n_bootstrap)
    else:
        seed = args.seed
        n_bootstrap = args.n_bootstrap

    print(f"Bootstrap seed: {seed}, n_bootstrap: {n_bootstrap}")
    print("NOTE: Run this script after preprocessing WESAD/SWELL-KW data.")
    print("Refer to data_instructions/ for dataset download and preparation.")


if __name__ == "__main__":
    main()
