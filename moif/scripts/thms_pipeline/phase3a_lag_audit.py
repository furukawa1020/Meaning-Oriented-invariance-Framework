import os
import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr
from scipy.interpolate import interp1d
import neurokit2 as nk

# Guard
if os.environ.get("ALLOW_REAL_DATA_PHASE_3A_LAG_AUDIT") != "1":
    print("FATAL: Phase 3A is locked.")
    sys.exit(1)

CASE_DIR = Path("C:/Projects/Meaning-Oriented invariance Framework/moif/data/case")
OUT_DIR = "phase3a_results"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Feature & Label Extraction ───────────────────────────────────────────────
def get_subject_data(sub_id):
    p_path = CASE_DIR / 'data' / 'interpolated' / 'physiological' / f"{sub_id}.csv"
    a_path = CASE_DIR / 'data' / 'interpolated' / 'annotations' / f"{sub_id}.csv"
    if not p_path.exists() or not a_path.exists(): return None
    df_p = pd.read_csv(p_path)
    df_a = pd.read_csv(a_path)
    
    tgt_fs = 100
    # Physio is at 1000Hz (daqtime in ms)
    t_p = df_p['daqtime'].values / 1000.0
    t_tgt = np.linspace(0, t_p[-1], int(t_p[-1] * tgt_fs), endpoint=False)
    
    # Annotations are at 20Hz (jstime in ms)
    t_a = df_a['jstime'].values / 1000.0
    
    # Clean & Resample Features
    eda = nk.eda_clean(df_p['gsr'].values, sampling_rate=1000)
    eda_100 = interp1d(t_p, eda, fill_value="extrapolate")(t_tgt)
    eda_phasic = nk.eda_phasic(eda_100, sampling_rate=tgt_fs)['EDA_Phasic'].values
    
    ecg = nk.ecg_clean(df_p['ecg'].values, sampling_rate=1000)
    rpeaks = nk.ecg_peaks(ecg, sampling_rate=1000)[1]['ECG_R_Peaks']
    rri_ms = np.diff(rpeaks)/1000.0*1000 # Correct: diff in indices / fs * 1000
    rri_t  = rpeaks[1:] / 1000.0
    rri_100 = interp1d(rri_t, rri_ms, kind='cubic', fill_value="extrapolate")(t_tgt)
    
    # Resample Labels
    # Convert 1-9 to -1 to 1? (Usually 1-9 in CASE)
    def rescale(v): return (v - 5.0) / 4.0
    arousal = interp1d(t_a, rescale(df_a['arousal'].values), fill_value="extrapolate")(t_tgt)
    valence = interp1d(t_a, rescale(df_a['valence'].values), fill_value="extrapolate")(t_tgt)
    video = interp1d(t_p, df_p['video'].values, kind='nearest', fill_value="extrapolate")(t_tgt)
    
    # 1s Windowing
    ws = 100
    nw = len(t_tgt) // ws
    def win(a): return a[:nw*ws].reshape(nw, ws).mean(axis=1)
    
    df_win = pd.DataFrame({
        'EDA_Phasic': win(eda_phasic), 'HRV_RRI': win(rri_100),
        'Arousal': win(arousal), 'Valence': win(valence),
        'video_id': win(video).astype(int)
    })
    return df_win

# ── Lagged Correlation Logic ─────────────────────────────────────────────────
def compute_lagged_corr(x, y, lag):
    """physiology(x) at t vs label(y) at t+lag."""
    if lag == 0:
        return spearmanr(x, y)[0]
    elif lag > 0:
        return spearmanr(x[:-lag], y[lag:])[0] if len(x) > lag else np.nan
    else:
        al = abs(lag)
        return spearmanr(x[al:], y[:-al])[0] if len(x) > al else np.nan

def run_lag_audit():
    lags = np.arange(-60, 61, 5) # -60s to +60s
    results = []
    
    for i in range(1, 31):
        sid = f"sub_{i}"
        print(f"Auditing CASE {sid}...")
        df = get_subject_data(sid)
        if df is None: continue
        
        for vid in df['video_id'].unique():
            if vid == 0: continue
            df_v = df[df['video_id'] == vid]
            
            for feat in ['EDA_Phasic', 'HRV_RRI']:
                for label in ['Arousal', 'Valence']:
                    x = df_v[feat].values
                    y = df_v[label].values
                    if len(x) < 70: continue # Min 70s per video
                    
                    # 1. True Lag Curve
                    true_corrs = {l: compute_lagged_corr(x, y, l) for l in lags}
                    
                    # 2. Circular Shift Control (Random shift between 10s and len-10s)
                    shift = np.random.randint(10, len(y) - 10)
                    y_shuffled = np.roll(y, shift)
                    ctrl_corrs = {l: compute_lagged_corr(x, y_shuffled, l) for l in lags}
                    
                    # Find best lag
                    best_lag = max(true_corrs, key=lambda l: abs(true_corrs[l]))
                    
                    results.append({
                        'subject_id': sid, 'video_id': vid, 'feat': feat, 'label': label,
                        'best_lag': int(best_lag), 'max_corr': true_corrs[best_lag],
                        'zero_lag_corr': true_corrs[0],
                        'ctrl_max_corr': max([abs(v) for v in ctrl_corrs.values()]),
                        'lag_curve_json': json.dumps({str(k): v for k, v in true_corrs.items()})
                    })
    
    pd.DataFrame(results).to_csv(os.path.join(OUT_DIR, "case_lag_correlation_report.csv"), index=False)
    print("CASE Lag Map generated.")

if __name__ == "__main__":
    import json
    print("Starting Phase 3A: CASE Lag Map & Control Audit...")
    run_lag_audit()
