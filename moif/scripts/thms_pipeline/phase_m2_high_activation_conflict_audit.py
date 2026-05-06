import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import neurokit2 as nk
import json

# Guard
if os.environ.get("ALLOW_REAL_DATA_PHASE_M2_AUDIT") != "1":
    print("FATAL: Phase M2 is locked.")
    sys.exit(1)

CASE_DIR = Path("C:/Projects/Meaning-Oriented invariance Framework/moif/data/case")
OUT_DIR = "phase_m_results"
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
    res_df = pd.DataFrame({
        'sub_id': sub_id, 'EDA_Phasic': win(eda_phasic), 'HRV_RRI': win(rri_100),
        'Arousal': win(arousal), 'Valence': win(valence),
        'video_id': win(video).astype(int), 'time_s': np.arange(nw)
    })
    scaler = StandardScaler()
    res_df[['EDA_Phasic', 'HRV_RRI']] = scaler.fit_transform(res_df[['EDA_Phasic', 'HRV_RRI']])
    return res_df[res_df['video_id'] != 0].copy()

# ── Audit ───────────────────────────────────────────────────────────────────
def run_conflict_audit():
    all_data = []
    for i in range(1, 16):
        df = get_subject_data(f"sub_{i}")
        if df is not None: all_data.append(df)
    full_df = pd.concat(all_data).reset_index(drop=True)
    
    # Define labels per subject thresholds
    processed_list = []
    for sid in full_df['sub_id'].unique():
        sub_df = full_df[full_df['sub_id'] == sid].copy()
        a_hi = np.percentile(sub_df['Arousal'], 70)
        a_lo = np.percentile(sub_df['Arousal'], 30)
        v_hi = np.percentile(sub_df['Valence'], 70)
        v_lo = np.percentile(sub_df['Valence'], 30)
        
        sub_df['is_HA'] = sub_df['Arousal'] > a_hi
        sub_df['is_LA'] = sub_df['Arousal'] < a_lo
        sub_df['is_PV'] = sub_df['Valence'] > v_hi
        sub_df['is_NV'] = sub_df['Valence'] < v_lo
        processed_list.append(sub_df)
    full_df = pd.concat(processed_list).reset_index(drop=True)
    
    feats = full_df[['EDA_Phasic', 'HRV_RRI']].values
    k = 50
    nn = NearestNeighbors(n_neighbors=k + 100, metric='euclidean')
    nn.fit(feats)
    distances, indices = nn.kneighbors(feats)
    
    results = []
    for i in range(len(full_df)):
        row = full_df.iloc[i]
        # Only audit High Arousal samples for Task C, and HA/LA for Task A
        if not (row['is_HA'] or row['is_LA']): continue
        
        target_sub = row['sub_id']
        target_time = row['time_s']
        idx_set = indices[i]
        valid = []
        for ni in idx_set:
            if ni == i: continue
            n_row = full_df.iloc[ni]
            if n_row['sub_id'] == target_sub and abs(n_row['time_s'] - target_time) < 30: continue
            valid.append(ni)
            if len(valid) >= k: break
        if len(valid) < k: continue
        
        n_df = full_df.iloc[valid]
        
        # 1. Activation Conflict (Arousal Hi vs Lo)
        # Only if current sample is HA or LA
        if row['is_HA'] or row['is_LA']:
            self_cat = 1 if row['is_HA'] else 0
            n_cats = n_df.apply(lambda r: 1 if r['is_HA'] else (0 if r['is_LA'] else -1), axis=1)
            n_cats = n_cats[n_cats != -1]
            if len(n_cats) > 0:
                act_conflict = (n_cats != self_cat).mean()
            else: act_conflict = np.nan
        else: act_conflict = np.nan

        # 2. Meaning Conflict (HA-PV vs HA-NV)
        # Only if current sample is HA and (PV or NV)
        if row['is_HA'] and (row['is_PV'] or row['is_NV']):
            self_v_cat = 1 if row['is_PV'] else 0
            # Neighbors that are HA AND (PV or NV)
            n_v_cats = n_df.apply(lambda r: 1 if (r['is_HA'] and r['is_PV']) else (0 if (r['is_HA'] and r['is_NV']) else -1), axis=1)
            n_v_cats = n_v_cats[n_v_cats != -1]
            if len(n_v_cats) > 0:
                mean_conflict = (n_v_cats != self_v_cat).mean()
            else: mean_conflict = np.nan
        else: mean_conflict = np.nan
        
        results.append({'act_conflict': act_conflict, 'mean_conflict': mean_conflict})
        
    res_df = pd.DataFrame(results)
    report = {
        "mean_activation_conflict": float(res_df['act_conflict'].dropna().mean()),
        "mean_meaning_conflict": float(res_df['mean_conflict'].dropna().mean()),
        "gap": float(res_df['mean_conflict'].dropna().mean() - res_df['act_conflict'].dropna().mean()),
        "gate_m2_3_pass": bool((res_df['mean_conflict'].dropna().mean() - res_df['act_conflict'].dropna().mean()) >= 0.10)
    }
    
    with open(os.path.join(OUT_DIR, "phase_m2_gate_report.json"), "w") as f:
        json.dump(report, f, indent=4)
    print("Phase M2 Gate Report Generated.")

if __name__ == "__main__":
    run_conflict_audit()
