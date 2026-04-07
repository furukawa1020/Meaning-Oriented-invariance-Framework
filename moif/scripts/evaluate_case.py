import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append('.')

from moif.loaders.case import load_case
from moif.signal.instantaneous import extract_instantaneous_features

case_raw_csv_path = "results/case_100hz_instantaneous_raw.csv"

def prepare_case_data():
    if os.path.exists(case_raw_csv_path):
        print(f"File {case_raw_csv_path} already exists. Loading it...")
        return pd.read_csv(case_raw_csv_path)

    print("Loading CASE dataset (1000Hz -> 100Hz)...")
    # Load raw interpolated dataframe
    df_raw = load_case('data/case')
    if df_raw.empty:
        print("Error: CASE data not loaded correctly or directory not found.")
        sys.exit(1)
        
    print("Applying Continuous Instantaneous Extraction (CWT & cvxEDA) per subject...")
    target_fs = 100
    dfs_extracted = []
    
    subjects = df_raw['subject_id'].unique()
    for subj in subjects:
        print(f"  Extracting features for subject {subj}...")
        df_subj = df_raw[df_raw['subject_id'] == subj].copy()
        
        # In CASE loader, ecg and gsr were already resampled to 100Hz in the df_raw.
        # It's better to use these 100Hz signals directly as input if extract_instantaneous_features supports it.
        # Let's check: extract_instantaneous_features(ecg, eda, fs_ecg, fs_eda, target_fs=100)
        # So we pass fs_ecg=100, fs_eda=100
        ecg_signal = df_subj['ECG'].values
        eda_signal = df_subj['GSR'].values
        
        try:
            df_feats = extract_instantaneous_features(
                ecg_signal, eda_signal, fs_ecg=target_fs, fs_eda=target_fs, target_fs=target_fs
            )
        except Exception as e:
            print(f"    Failed to extract for {subj}: {e}")
            continue
            
        # extract_instantaneous_features returns a DataFrame with 'timestamp', EDA_*, HRV_*
        # We need to map the labels back
        df_feats['subject_id'] = subj
        df_feats['label'] = df_subj['label'].values[:len(df_feats)] # Align lengths
        if 'valence' in df_subj.columns:
            df_feats['valence'] = df_subj['valence'].values[:len(df_feats)]
            df_feats['arousal'] = df_subj['arousal'].values[:len(df_feats)]
            
        dfs_extracted.append(df_feats)
        
    df_final = pd.concat(dfs_extracted, ignore_index=True)
    df_final = df_final.dropna(subset=['label', 'HRV_Inst_HF', 'EDA_Tonic'])
    
    print(f"Saving extracted CASE dataset to {case_raw_csv_path}...")
    os.makedirs('results', exist_ok=True)
    df_final.to_csv(case_raw_csv_path, index=False)
    return df_final

if __name__ == "__main__":
    df_case = prepare_case_data()
    print("CASE extraction complete.")
    
    # Now run the baseline evaluation on CASE using the same logic
    # We can just import and call process_subject from evaluate_baselines
    # But wait, evaluate_baselines.py doesn't export process_subject cleanly if we run it as main.
    # Actually we can just run evaluate_baselines by passing the csv path as an argument.
