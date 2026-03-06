import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
import warnings

warnings.filterwarnings('ignore')

LABEL_MAP = {
    1: 'baseline',
    2: 'stress',
    3: 'amusement',
    4: 'meditation'
}

def _compute_hr(ecg_signal: np.ndarray, fs: int = 700) -> np.ndarray:
    """Compute instantaneous HR from ECG in 1Hz resolution using RR intervals."""
    ecg = ecg_signal.flatten()
    q90 = np.percentile(ecg, 90)
    # Using height > 0.5 * 90th percentile to avoid T-waves, distance > 300ms (max 200 bpm)
    peaks, _ = find_peaks(ecg, distance=int(fs*0.3), height=q90*0.5)
    
    total_sec = len(ecg) // fs
    times_target = np.arange(total_sec)
    
    if len(peaks) < 2:
        return np.full(total_sec, np.nan)
        
    peak_times = peaks / fs
    rr = np.diff(peak_times)
    hr_vals = 60.0 / rr
    peak_times = peak_times[1:]
    
    # Interpolate to 1Hz
    hr_interp = np.interp(times_target, peak_times, hr_vals)
    return hr_interp

def _align_eda(eda_signal: np.ndarray, fs: int = 4, target_length: int = 0) -> np.ndarray:
    """Downsample EMA to 1Hz using block averaging."""
    eda = eda_signal.flatten()
    total_sec = len(eda) // fs
    eda_1hz = np.array([np.mean(eda[i*fs : (i+1)*fs]) for i in range(total_sec)])
    if target_length > 0:
        if len(eda_1hz) < target_length:
            eda_1hz = np.pad(eda_1hz, (0, target_length - len(eda_1hz)), 'edge')
        else:
            eda_1hz = eda_1hz[:target_length]
    return eda_1hz

def _align_labels(labels: np.ndarray, fs: int = 700, target_length: int = 0) -> np.ndarray:
    labels = labels.flatten()
    total_sec = len(labels) // fs
    label_1hz = np.zeros(total_sec, dtype=int)
    for i in range(total_sec):
        chunk = labels[i*fs : (i+1)*fs]
        val, counts = np.unique(chunk, return_counts=True)
        label_1hz[i] = val[np.argmax(counts)]
    
    if target_length > 0:
        if len(label_1hz) < target_length:
            label_1hz = np.pad(label_1hz, (0, target_length - len(label_1hz)), 'edge')
        else:
            label_1hz = label_1hz[:target_length]
    return label_1hz

def load_wesad(data_dir: str | Path, signals: list[str]) -> pd.DataFrame:
    """
    Load WESAD finding S*.pkl files.
    """
    root = Path(data_dir)
    records = []
    
    pkl_files = list(root.rglob("*.pkl"))
    if not pkl_files:
        raise FileNotFoundError(f"No .pkl files found in {root}")
        
    for p_path in pkl_files:
        with open(p_path, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
            
        subj_id = data['subject']
        lbls = data['label']
        total_sec = len(lbls) // 700
        
        # 1Hz label stream
        label_1hz = _align_labels(lbls, fs=700, target_length=total_sec)
        
        hr_1hz = None
        eda_1hz = None
        
        if "HR" in signals:
            ecg = data['signal']['chest']['ECG']
            hr_1hz = _compute_hr(ecg, fs=700)
            if len(hr_1hz) < total_sec:
                hr_1hz = np.pad(hr_1hz, (0, total_sec - len(hr_1hz)), 'edge')
            else:
                hr_1hz = hr_1hz[:total_sec]
                
        if "EDA" in signals:
            eda_e4 = data['signal']['wrist']['EDA']
            eda_1hz = _align_eda(eda_e4, fs=4, target_length=total_sec)
            
        # Compile directly to DataFrame records
        for i in range(total_sec):
            label_id = label_1hz[i]
            if label_id not in LABEL_MAP:
                continue
                
            task_name = LABEL_MAP[label_id]
            
            if "HR" in signals:
                records.append({
                    "timestamp": float(i),
                    "subject_id": subj_id,
                    "session_id": "1",
                    "task": task_name,
                    "signal_name": "HR",
                    "value": float(hr_1hz[i]),
                    "label": task_name
                })
            
            if "EDA" in signals:
                records.append({
                    "timestamp": float(i),
                    "subject_id": subj_id,
                    "session_id": "1",
                    "task": task_name,
                    "signal_name": "EDA",
                    "value": float(eda_1hz[i]),
                    "label": task_name
                })
                
    df = pd.DataFrame(records)
    return df
