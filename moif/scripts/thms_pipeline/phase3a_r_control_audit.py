import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr, ks_2samp
from scipy.interpolate import interp1d
import neurokit2 as nk
import json

# Guard
if os.environ.get("ALLOW_REAL_DATA_PHASE_3A_R_AUDIT") != "1":
    print("FATAL: Phase 3A-R is locked.")
    sys.exit(1)

CASE_DIR = Path("C:/Projects/Meaning-Oriented invariance Framework/moif/data/case")
OUT_DIR = "phase3a_r_results"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Loader (Reuse from 3A) ──────────────────────────────────────────────────
def get_subject_data(sub_id):
    p_path = CASE_DIR / 'data' / 'interpolated' / 'physiological' / f"{sub_id}.csv"
    a_path = CASE_DIR / 'data' / 'interpolated' / 'annotations' / f"{sub_id}.csv"
    if not p_path.exists() or not a_path.exists(): return None
    df_p = pd.read_csv(p_path)
    df_a = pd.read_csv(a_path)
    
    tgt_fs = 100
    t_p = df_p['daqtime'].values / 1000.0
    t_tgt = np.linspace(0, t_p[-1], int(t_p[-1] * tgt_fs), endpoint=False)
    t_a = df_a['jstime'].values / 1000.0
    
    eda = nk.eda_clean(df_p['gsr'].values, sampling_rate=1000)
    eda_100 = interp1d(t_p, eda, fill_value="extrapolate")(t_tgt)
    eda_phasic = nk.eda_phasic(eda_100, sampling_rate=tgt_fs)['EDA_Phasic'].values
    
    ecg = nk.ecg_clean(df_p['ecg'].values, sampling_rate=1000)
    rpeaks = nk.ecg_peaks(ecg, sampling_rate=1000)[1]['ECG_R_Peaks']
    rri_ms = np.diff(rpeaks)/1000.0*1000
    rri_100 = interp1d(rpeaks[1:] / 1000.0, rri_ms, kind='cubic', fill_value="extrapolate")(t_tgt)
    
    def rescale(v): return (v - 5.0) / 4.0
    arousal = interp1d(t_a, rescale(df_a['arousal'].values), fill_value="extrapolate")(t_tgt)
    valence = interp1d(t_a, rescale(df_a['valence'].values), fill_value="extrapolate")(t_tgt)
    video = interp1d(t_p, df_p['video'].values, kind='nearest', fill_value="extrapolate")(t_tgt)
    
    ws = 100
    nw = len(t_tgt) // ws
    def win(a): return a[:nw*ws].reshape(nw, ws).mean(axis=1)
    
    return pd.DataFrame({
        'EDA_Phasic': win(eda_phasic), 'HRV_RRI': win(rri_100),
        'Arousal': win(arousal), 'Valence': win(valence),
        'video_id': win(video).astype(int)
    })

# ── Audit Logic ─────────────────────────────────────────────────────────────
def compute_lagged_corr(x, y, lag):
    if lag == 0: return spearmanr(x, y)[0]
    elif lag > 0: return spearmanr(x[:-lag], y[lag:])[0] if len(x) > lag else np.nan
    else:
        al = abs(lag)
        return spearmanr(x[al:], y[:-al])[0] if len(x) > al else np.nan

def run_red_team_audit():
    lags = np.arange(-60, 61, 5)
    results = []
    
    # Process 10 subjects for the Red-Team Audit
    for i in range(1, 11):
        sid = f"sub_{i}"
        print(f"Red-Team Audit: CASE {sid}...")
        df = get_subject_data(sid)
        if df is None: continue
        
        for vid in df['video_id'].unique():
            if vid == 0: continue
            df_v = df[df['video_id'] == vid]
            
            for feat in ['EDA_Phasic', 'HRV_RRI']:
                for label in ['Arousal', 'Valence']:
                    x = df_v[feat].values
                    y = df_v[label].values
                    if len(x) < 70: continue
                    
                    # 1. True Lag
                    true_corrs = {l: compute_lagged_corr(x, y, l) for l in lags}
                    best_l = max(true_corrs, key=lambda l: abs(true_corrs[l]))
                    
                    # 2. Block Shuffle Control (10s blocks)
                    block_size = 10
                    n_blocks = len(y) // block_size
                    indices = np.arange(n_blocks)
                    np.random.shuffle(indices)
                    y_shuffled = np.concatenate([y[i*block_size:(i+1)*block_size] for i in indices])
                    if len(y_shuffled) < len(y): y_shuffled = np.pad(y_shuffled, (0, len(y)-len(y_shuffled)), 'edge')
                    bs_corrs = {l: compute_lagged_corr(x, y_shuffled, l) for l in lags}
                    
                    # 3. Time Reversal Control
                    y_rev = y[::-1]
                    rev_corrs = {l: compute_lagged_corr(x, y_rev, l) for l in lags}
                    
                    # 4. First Difference Robustness
                    dx, dy = np.diff(x), np.diff(y)
                    diff_corrs = {l: compute_lagged_corr(dx, dy, l) for l in lags}
                    best_diff_l = max(diff_corrs, key=lambda l: abs(diff_corrs[l]))
                    
                    results.append({
                        'subject': sid, 'video': vid, 'feat': feat, 'label': label,
                        'true_best_lag': int(best_l), 'true_max_corr': true_corrs[best_l],
                        'bs_max_corr': max([abs(v) for v in bs_corrs.values()]),
                        'rev_max_corr': max([abs(v) for v in rev_corrs.values()]),
                        'diff_best_lag': int(best_diff_l), 'diff_max_corr': diff_corrs[best_diff_l]
                    })
                    
    pd.DataFrame(results).to_csv(os.path.join(OUT_DIR, "phase3a_r_audit_results.csv"), index=False)
    print("Red-Team Audit complete.")

if __name__ == "__main__":
    print("Starting Phase 3A-R: Control Validity Red-Team Audit...")
    run_red_team_audit()
