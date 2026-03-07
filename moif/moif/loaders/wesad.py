"""
Advanced WESAD Dataloader using NeuroKit2
Slices physiological signals into sliding windows (e.g., 60s window, 10s stride)
and extracts domain-specific features (HRV, Phasic/Tonic EDA).
"""
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import warnings

from moif.signal.advanced_features import extract_window_features

warnings.filterwarnings('ignore')

LABEL_MAP = {
    1: 'baseline',
    2: 'stress',
    3: 'amusement',
    4: 'meditation'
}

def load_wesad(data_dir: str | Path, window_size_sec: int = 60, stride_sec: int = 10) -> pd.DataFrame:
    """
    Load WESAD finding S*.pkl files.
    Applies sliding window feature extraction using neurokit2.
    """
    root = Path(data_dir)
    records = []
    
    pkl_files = list(root.rglob("*.pkl"))
    if not pkl_files:
        raise FileNotFoundError(f"No .pkl files found in {root}")
        
    # Limit to 3 subjects for faster theoretical validation in this run
    pkl_files = sorted(pkl_files)[:3]
        
    fs_ecg = 700
    fs_eda = 4
    
    # Pre-calculate window samples
    # We will slide across the dataset using stride_sec, keeping window_size_sec of data
    
    for p_path in pkl_files:
        print(f"Processing {p_path.name}...")
        with open(p_path, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
            
        subj_id = data['subject']
        lbls = data['label'].flatten() # 700Hz
        
        # Total duration in seconds
        total_sec = len(lbls) // 700
        
        ecg_chest = data['signal']['chest']['ECG'].flatten()
        eda_wrist = data['signal']['wrist']['EDA'].flatten()
        
        # Sliding window
        current_start_sec = 0
        while current_start_sec + window_size_sec <= total_sec:
            # Extract labels for this string
            start_lbl_idx = current_start_sec * 700
            end_lbl_idx = (current_start_sec + window_size_sec) * 700
            chunk_lbls = lbls[start_lbl_idx : end_lbl_idx]
            
            # Determine dominant label in window
            val, counts = np.unique(chunk_lbls, return_counts=True)
            dominant_label = val[np.argmax(counts)]
            
            # Only keep specified labels (Baseline, Stress, Amusement, Meditation)
            if dominant_label not in LABEL_MAP:
                current_start_sec += stride_sec
                continue
                
            task_name = LABEL_MAP[dominant_label]
            
            # Extract signal chunks
            start_ecg_idx = current_start_sec * fs_ecg
            end_ecg_idx = (current_start_sec + window_size_sec) * fs_ecg
            chunk_ecg = ecg_chest[start_ecg_idx : end_ecg_idx]
            
            start_eda_idx = current_start_sec * fs_eda
            end_eda_idx = (current_start_sec + window_size_sec) * fs_eda
            chunk_eda = eda_wrist[start_eda_idx : end_eda_idx]
            
            # Extract features (using our new neurokit2 module)
            feats = extract_window_features(chunk_ecg, chunk_eda, fs_ecg=fs_ecg, fs_eda=fs_eda)
            
            # Store record
            record = {
                "subject_id": subj_id,
                "session_id": "1",
                "timestamp_start": current_start_sec,
                "timestamp_end": current_start_sec + window_size_sec,
                "label": task_name
            }
            # Unpack features
            for k, v in feats.items():
                record[k] = v
                
            records.append(record)
            
            current_start_sec += stride_sec
            
    df = pd.DataFrame(records)
    # Drop rows where feature extraction failed across the board
    df = df.dropna(subset=['HRV_RMSSD', 'EDA_Tonic_Mean'], how='any')
    return df
