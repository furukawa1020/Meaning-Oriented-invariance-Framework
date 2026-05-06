import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from scipy.stats import entropy
import neurokit2 as nk
import json

# Guard
if os.environ.get("ALLOW_REAL_DATA_PHASE_M1_R_AUDIT") != "1":
    print("FATAL: Phase M1-R Finalizer is locked.")
    sys.exit(1)

CASE_DIR = Path("C:/Projects/Meaning-Oriented invariance Framework/moif/data/case")
OUT_DIR = "phase_m_r_results"
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

# ── Stats ───────────────────────────────────────────────────────────────────
def run_final_stats():
    all_data = []
    for i in range(1, 16):
        df = get_subject_data(f"sub_{i}")
        if df is not None: all_data.append(df)
    full_df = pd.concat(all_data).reset_index(drop=True)
    
    def discretize(v): return np.digitize(v, bins=[-0.33, 0.33])
    full_df['A_cat'] = discretize(full_df['Arousal'].values)
    full_df['V_cat'] = discretize(full_df['Valence'].values)
    
    h_a_global = entropy(full_df['A_cat'].value_counts(normalize=True), base=2)
    h_v_global = entropy(full_df['V_cat'].value_counts(normalize=True), base=2)
    
    k = 50
    feats = full_df[['EDA_Phasic', 'HRV_RRI']].values
    nn = NearestNeighbors(n_neighbors=k + 100, metric='euclidean')
    nn.fit(feats)
    distances, indices = nn.kneighbors(feats)
    
    local_ents_a = []
    local_ents_v = []
    
    for i in range(len(full_df)):
        target_sub = full_df.iloc[i]['sub_id']
        target_time = full_df.iloc[i]['time_s']
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
        def ent(arr): return entropy(arr.value_counts(normalize=True), base=2)
        local_ents_a.append(ent(n_df['A_cat']))
        local_ents_v.append(ent(n_df['V_cat']))
        
    local_ents_a = np.array(local_ents_a)
    local_ents_v = np.array(local_ents_v)
    
    # Bootstrap
    n_boot = 200
    boot_deltas = []
    boot_norms = []
    for _ in range(n_boot):
        idx = np.random.choice(len(local_ents_a), len(local_ents_a), replace=True)
        m_a = np.mean(local_ents_a[idx])
        m_v = np.mean(local_ents_v[idx])
        boot_deltas.append(m_v - m_a)
        boot_norms.append((m_v/h_v_global) - (m_a/h_a_global))
        
    final_report = {
        "mean_delta": float(np.mean(local_ents_v - local_ents_a)),
        "delta_95_ci": [float(np.percentile(boot_deltas, 2.5)), float(np.percentile(boot_deltas, 97.5))],
        "mean_norm_delta": float(np.mean(boot_norms)),
        "norm_delta_95_ci": [float(np.percentile(boot_norms, 2.5)), float(np.percentile(boot_norms, 97.5))],
        "global_H": {"A": h_a_global, "V": h_v_global},
        "time_adj_exclusion": "±30s enforced"
    }
    
    with open(os.path.join(OUT_DIR, "phase_m1r_final_stats.json"), "w") as f:
        json.dump(final_report, f, indent=4)
    print("Phase M1-R Final Stats Generated.")

if __name__ == "__main__":
    run_final_stats()
