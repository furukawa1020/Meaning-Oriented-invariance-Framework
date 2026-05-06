import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.interpolate import interp1d
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import neurokit2 as nk
import json

# Guard
if os.environ.get("ALLOW_REAL_DATA_PHASE_U2_AUDIT") != "1":
    print("FATAL: Phase U2 is locked.")
    sys.exit(1)

CASE_DIR = Path("C:/Projects/Meaning-Oriented invariance Framework/moif/data/case")
OUT_DIR = "phase_u_results"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Loader ──────────────────────────────────────────────────────────────────
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
        'sub_id': sub_id,
        'EDA_Phasic': win(eda_phasic), 'HRV_RRI': win(rri_100),
        'Arousal': win(arousal), 'Valence': win(valence),
        'video_id': win(video).astype(int)
    })

# ── Evaluation ──────────────────────────────────────────────────────────────
def run_diagnostic_audit():
    all_data = []
    for i in range(1, 31):
        sid = f"sub_{i}"
        df = get_subject_data(sid)
        if df is not None: all_data.append(df)
    full_df = pd.concat(all_data)
    
    # Exclude inter-video baseline
    full_df = full_df[full_df['video_id'] != 0]

    def get_loso_scores(df_task):
        subs = df_task['sub_id'].unique()
        scores = []
        for test_sub in subs:
            train = df_task[df_task['sub_id'] != test_sub]
            test = df_task[df_task['sub_id'] == test_sub]
            if len(test['label'].unique()) < 2: continue
            
            clf = LogisticRegression()
            feats = ['EDA_Phasic', 'HRV_RRI']
            clf.fit(train[feats], train['label'])
            probs = clf.predict_proba(test[feats])[:, 1]
            scores.append(roc_auc_score(test['label'], probs))
        return scores

    results = {}
    
    # Task A: HA vs LA
    print("Task A: HA vs LA...")
    # Per-subject thresholds
    task_a_data = []
    for sid in full_df['sub_id'].unique():
        sub_df = full_df[full_df['sub_id'] == sid]
        hi = np.percentile(sub_df['Arousal'], 70)
        lo = np.percentile(sub_df['Arousal'], 30)
        tdf = sub_df[(sub_df['Arousal'] > hi) | (sub_df['Arousal'] < lo)].copy()
        tdf['label'] = (tdf['Arousal'] > hi).astype(int)
        task_a_data.append(tdf)
    results['Task_A'] = np.mean(get_loso_scores(pd.concat(task_a_data)))

    # Task B: PV vs NV
    print("Task B: PV vs NV...")
    task_b_data = []
    for sid in full_df['sub_id'].unique():
        sub_df = full_df[full_df['sub_id'] == sid]
        hi = np.percentile(sub_df['Valence'], 70)
        lo = np.percentile(sub_df['Valence'], 30)
        tdf = sub_df[(sub_df['Valence'] > hi) | (sub_df['Valence'] < lo)].copy()
        tdf['label'] = (tdf['Valence'] > hi).astype(int)
        task_b_data.append(tdf)
    results['Task_B'] = np.mean(get_loso_scores(pd.concat(task_b_data)))

    # Task C: HA-Pos vs HA-Neg
    print("Task C: HA-Pos vs HA-Neg...")
    task_c_data = []
    for sid in full_df['sub_id'].unique():
        sub_df = full_df[full_df['sub_id'] == sid]
        a_hi = np.percentile(sub_df['Arousal'], 70)
        v_hi = np.percentile(sub_df['Valence'], 70)
        v_lo = np.percentile(sub_df['Valence'], 30)
        # Meaning ambiguity WITHIN high activation
        tdf = sub_df[(sub_df['Arousal'] > a_hi) & ((sub_df['Valence'] > v_hi) | (sub_df['Valence'] < v_lo))].copy()
        tdf['label'] = (tdf['Valence'] > v_hi).astype(int)
        task_c_data.append(tdf)
    results['Task_C'] = np.mean(get_loso_scores(pd.concat(task_c_data)))

    results['delta_A_C'] = results['Task_A'] - results['Task_C']
    
    with open(os.path.join(OUT_DIR, "phase_u2_diagnostic_report.json"), "w") as f:
        json.dump(results, f, indent=4)
    print("Phase U2 Diagnostic Report generated.")

if __name__ == "__main__":
    run_diagnostic_audit()
