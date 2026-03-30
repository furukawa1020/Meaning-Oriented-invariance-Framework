"""
CASE (Continuously Annotated Signals of Emotion) Dataloader.
Data: 30 subjects, 1000Hz raw, interpolated to 20Hz annotations? 
We will upsample everything to 100Hz as per MOIF standard.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.interpolate import interp1d
import warnings

warnings.filterwarnings('ignore')

# Mapping for CASE video IDs to labels
# 1: scary-1, 2: scary-2, 3: amusing-1, 4: amusing-2, 
# 5: boring-1, 6: boring-2, 7: relaxed-1, 8: relaxed-2
# 10: bluVid (Baseline)
CASE_LABEL_MAP = {
    10: 'baseline',
    1: 'stress',  # scary -> stress equivalent
    2: 'stress',
    3: 'amusement',
    4: 'amusement',
    7: 'meditation', # relaxed -> meditation equivalent
    8: 'meditation'
}

def load_case(data_dir: str | Path, subj_ids: list[str] | None = None) -> pd.DataFrame:
    """
    Load CASE dataset files from the interpolated directory.
    Synchronizes 1000Hz physiological signals with 20Hz annotations.
    """
    root = Path(data_dir)
    phys_dir = root / 'data' / 'interpolated' / 'physiological'
    anno_dir = root / 'data' / 'interpolated' / 'annotations'
    
    if not phys_dir.exists() or not anno_dir.exists():
        raise FileNotFoundError(f"Missing data directories in {root}")

    phys_files = sorted(list(phys_dir.glob("sub_*.csv")))
    
    if subj_ids:
        phys_files = [f for f in phys_files if f.stem in subj_ids]
        
    target_fs = 100
    dfs = []

    for p_path in phys_files:
        subj_id = p_path.stem  # e.g., 'sub_1'
        a_path = anno_dir / f"{subj_id}.csv"
        
        if not a_path.exists():
            print(f"Skipping {subj_id}: annotation file not found.")
            continue
            
        print(f"Loading CASE subject: {subj_id}")
        
        # 1. Load Physiological (1000Hz)
        df_p = pd.read_csv(p_path)
        # daqtime is ms
        t_p = df_p['daqtime'].values / 1000.0
        
        # 2. Load Annotations (20Hz)
        df_a = pd.read_csv(a_path)
        t_a = df_a['jstime'].values / 1000.0
        
        duration_sec = min(t_p[-1], t_a[-1])
        target_length = int(np.floor(duration_sec * target_fs))
        t_target = np.linspace(0, duration_sec, target_length, endpoint=False)

        # Resample key features to 100Hz
        df_resampled = pd.DataFrame({'timestamp': t_target})
        
        # Physiological (ECG, GSR, BVP, RSP, SKT)
        # Using 100Hz instantaneous features for HRV/EDA just like WESAD
        # We wrap the 100Hz extraction logic for CASE too
        # But for now, let's just get the raw values to see if things work
        for col in ['ecg', 'gsr', 'bvp', 'video']:
            if col in df_p.columns:
                # Use nearest for video ID to avoid float labels
                kind = 'nearest' if col == 'video' else 'cubic'
                f = interp1d(t_p, df_p[col].values, kind=kind, bounds_error=False, fill_value="extrapolate")
                df_resampled[col.upper()] = f(t_target)

        # Subjective Labels (Valence / Arousal)
        for col in ['valence', 'arousal']:
            if col in df_a.columns:
                f = interp1d(t_a, df_a[col].values, kind='linear', bounds_error=False, fill_value="extrapolate")
                df_resampled[col] = f(t_target)

        # Map Video IDs to semantic labels
        df_resampled['label'] = df_resampled['VIDEO'].map(CASE_LABEL_MAP)
        
        # Apply the same Instantaneous Feature Extraction as WESAD
        # This will be done in the evaluation script to stay modular
        df_resampled['subject_id'] = subj_id
        
        # Drop non-mapped segments (startVid, endVid, etc.)
        df_resampled = df_resampled.dropna(subset=['label'])
        
        dfs.append(df_resampled)

    if not dfs:
        return pd.DataFrame()
        
    df_final = pd.concat(dfs, ignore_index=True)
    return df_final
