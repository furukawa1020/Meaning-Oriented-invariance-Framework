"""
Advanced WESAD Dataloader using Instantaneous 100Hz physiological trackers.
"""
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import warnings
from scipy.interpolate import interp1d

from moif.signal.instantaneous import extract_instantaneous_features

warnings.filterwarnings('ignore')

LABEL_MAP = {
    1: 'baseline',
    2: 'stress',
    3: 'amusement',
    4: 'meditation'
}

def load_wesad(data_dir: str | Path) -> pd.DataFrame:
    """
    Load WESAD finding S*.pkl files.
    Applies continuous, instantaneous 100Hz feature extraction using CWT and exact models.
    """
    root = Path(data_dir)
    
    pkl_files = list(root.rglob("*.pkl"))
    if not pkl_files:
        raise FileNotFoundError(f"No .pkl files found in {root}")
        
    # Process all subjects for the rigorous theoretical validation
    pkl_files = sorted(pkl_files)
        
    fs_ecg = 700
    fs_eda = 4
    target_fs = 100
    
    dfs = []
    
    for p_path in pkl_files:
        print(f"Applying Continuous Instantaneous Extraction on {p_path.name}...")
        with open(p_path, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
            
        subj_id = data['subject']
        lbls_700hz = data['label'].flatten() # 700Hz
        ecg_chest = data['signal']['chest']['ECG'].flatten()
        eda_wrist = data['signal']['wrist']['EDA'].flatten()
        
        # 1. Extract 100Hz continuous features (This takes a moment)
        print(f"  Deconvoluting physiological features at {target_fs}Hz...")
        df_feats = extract_instantaneous_features(
            ecg_chest, eda_wrist, fs_ecg=fs_ecg, fs_eda=fs_eda, target_fs=target_fs
        )
        
        # 2. Resample the subjective labels from 700Hz to 100Hz (nearest neighbor)
        print("  Synchronizing subjective meaning labels...")
        t_orig_lbls = np.linspace(0, len(lbls_700hz)/700, len(lbls_700hz), endpoint=False)
        f_lbls = interp1d(t_orig_lbls, lbls_700hz, kind='nearest', bounds_error=False, fill_value="extrapolate")
        
        lbls_100hz = f_lbls(df_feats['timestamp'].values)
        
        df_feats['raw_label'] = lbls_100hz
        df_feats['subject_id'] = subj_id
        
        # Map labels to meaningful classes and drop NaN/unmapped
        df_feats['label'] = df_feats['raw_label'].map(LABEL_MAP)
        df_feats = df_feats.dropna(subset=['label', 'HRV_Inst_HF', 'EDA_Tonic'])
        
        dfs.append(df_feats)
        
    df_final = pd.concat(dfs, ignore_index=True)
    return df_final
