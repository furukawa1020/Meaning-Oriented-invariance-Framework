import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.interpolate import interp1d
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneOut
import neurokit2 as nk

# Guard
if os.environ.get("ALLOW_REAL_DATA_PHASE_U1_AUDIT") != "1":
    print("FATAL: Phase U1 is locked.")
    sys.exit(1)

CASE_DIR = Path("C:/Projects/Meaning-Oriented invariance Framework/moif/data/case")
OUT_DIR = "phase_u_results"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Loader (Standardized) ───────────────────────────────────────────────────
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
    
    # Features
    eda = nk.eda_clean(df_p['gsr'].values, sampling_rate=1000)
    eda_100 = interp1d(t_p, eda, fill_value="extrapolate")(t_tgt)
    eda_phasic = nk.eda_phasic(eda_100, sampling_rate=tgt_fs)['EDA_Phasic'].values
    
    ecg = nk.ecg_clean(df_p['ecg'].values, sampling_rate=1000)
    rpeaks = nk.ecg_peaks(ecg, sampling_rate=1000)[1]['ECG_R_Peaks']
    rri_ms = np.diff(rpeaks)/1000.0*1000
    rri_100 = interp1d(rpeaks[1:] / 1000.0, rri_ms, kind='cubic', fill_value="extrapolate")(t_tgt)
    
    # Labels (Rescaled to -1 to 1)
    def rescale(v): return (v - 5.0) / 4.0
    arousal = interp1d(t_a, rescale(df_a['arousal'].values), fill_value="extrapolate")(t_tgt)
    valence = interp1d(t_a, rescale(df_a['valence'].values), fill_value="extrapolate")(t_tgt)
    
    ws = 100
    nw = len(t_tgt) // ws
    def win(a): return a[:nw*ws].reshape(nw, ws).mean(axis=1)
    
    return pd.DataFrame({
        'sub_id': sub_id,
        'EDA_Phasic': win(eda_phasic), 'HRV_RRI': win(rri_100),
        'Arousal': win(arousal), 'Valence': win(valence)
    })

# ── Audit Logic ─────────────────────────────────────────────────────────────
def run_asymmetry_audit():
    all_data = []
    for i in range(1, 31):
        sid = f"sub_{i}"
        print(f"Loading CASE {sid}...")
        df = get_subject_data(sid)
        if df is not None: all_data.append(df)
    
    full_df = pd.concat(all_data)
    
    # 1. Define Binary Tasks
    # High vs Low Arousal
    ar_df = full_df[(full_df['Arousal'] > 0.33) | (full_df['Arousal'] < -0.33)].copy()
    ar_df['label'] = (ar_df['Arousal'] > 0.33).astype(int)
    
    # Positive vs Negative Valence
    va_df = full_df[(full_df['Valence'] > 0.33) | (full_df['Valence'] < -0.33)].copy()
    va_df['label'] = (va_df['Valence'] > 0.33).astype(int)
    
    def eval_task(df, name):
        subs = df['sub_id'].unique()
        scores = []
        for test_sub in subs:
            train_df = df[df['sub_id'] != test_sub]
            test_df = df[df['sub_id'] == test_sub]
            
            if len(test_df['label'].unique()) < 2: continue
            
            clf = LogisticRegression()
            feats = ['EDA_Phasic', 'HRV_RRI']
            clf.fit(train_df[feats], train_df['label'])
            probs = clf.predict_proba(test_df[feats])[:, 1]
            scores.append(roc_auc_score(test_df['label'], probs))
        return scores

    print("Evaluating Arousal Predictability...")
    ar_scores = eval_task(ar_df, "Arousal")
    
    print("Evaluating Valence Predictability...")
    va_scores = eval_task(va_df, "Valence")
    
    report = {
        "arousal_mean_auroc": float(np.mean(ar_scores)),
        "valence_mean_auroc": float(np.mean(va_scores)),
        "delta_auroc": float(np.mean(ar_scores) - np.mean(va_scores)),
        "gate_u1_pass": bool((np.mean(ar_scores) - np.mean(va_scores)) >= 0.05)
    }
    
    with open(os.path.join(OUT_DIR, "phase_u1_asymmetry_report.json"), "w") as f:
        json.dump(report, f, indent=4)
    print("Phase U1 Asymmetry Report generated.")

if __name__ == "__main__":
    import json
    run_asymmetry_audit()
