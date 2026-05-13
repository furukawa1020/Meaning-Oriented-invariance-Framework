"""
swell_aggregation.py
SWELL-KW protocol-level block aggregation script.

Aggregates minute-level physiological features and behavioral covariates
to 15-minute block-level units for the Resolution Audit case study.

Reference: LabelScope manuscript, Section IV-B
"""
import numpy as np
import pandas as pd
import json
from pathlib import Path


def aggregate_to_blocks(df, block_duration_min=15, window_duration_min=1):
    """
    Aggregate minute-level windows to block-level feature vectors.

    Args:
        df: DataFrame with columns [subject_id, condition, minute_idx, features..., covariate]
        block_duration_min: duration of each block in minutes
        window_duration_min: duration of each window in minutes

    Returns:
        df_block: aggregated block-level DataFrame
    """
    windows_per_block = block_duration_min // window_duration_min
    df_block_rows = []

    for (subject, condition), group in df.groupby(['subject_id', 'condition']):
        group = group.sort_values('minute_idx').reset_index(drop=True)
        n_blocks = len(group) // windows_per_block

        for b in range(n_blocks):
            block_data = group.iloc[b * windows_per_block:(b + 1) * windows_per_block]
            feature_cols = [c for c in df.columns
                            if c not in ['subject_id', 'condition', 'minute_idx', 'label', 'covariate']]
            block_features = block_data[feature_cols].mean().to_dict()
            block_covariate = block_data['covariate'].sum()  # aggregate interaction intensity
            block_label = block_data['label'].mode()[0]

            row = {'subject_id': subject, 'condition': condition, 'block_idx': b}
            row.update(block_features)
            row['covariate'] = block_covariate
            row['label'] = block_label
            df_block_rows.append(row)

    return pd.DataFrame(df_block_rows)


def compute_stressor_composite(df_block):
    """
    Average Time Pressure and Interruption blocks within subject to form
    one stressor-composite block for binary PA (Neutral vs. stressor-composite).

    Returns:
        df_binary: DataFrame with Neutral and stressor-composite observations
    """
    stressor_conditions = ['time_pressure', 'interruption']
    rows = []

    for subject, group in df_block.groupby('subject_id'):
        neutral = group[group['condition'] == 'neutral']
        stressor = group[group['condition'].isin(stressor_conditions)]

        if len(neutral) > 0:
            rows.append({'subject_id': subject, 'condition': 'neutral',
                         'label': 0, **neutral.drop(columns=['subject_id', 'condition', 'label']).mean().to_dict()})
        if len(stressor) > 0:
            rows.append({'subject_id': subject, 'condition': 'stressor_composite',
                         'label': 1, **stressor.drop(columns=['subject_id', 'condition', 'label']).mean().to_dict()})

    return pd.DataFrame(rows)


if __name__ == "__main__":
    with open("configs/preprocessing_config.json") as f:
        cfg = json.load(f)["swell_kw"]

    print(f"Block duration: {cfg['block_duration_min']} min")
    print(f"Behavioral covariate: {cfg['covariate']}")
    print("Run this script after preparing SWELL-KW data per data_instructions/.")
