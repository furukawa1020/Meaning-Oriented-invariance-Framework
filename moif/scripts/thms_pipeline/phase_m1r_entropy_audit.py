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
    print("FATAL: Phase M1-R is locked.")
    sys.exit(1)

CASE_DIR = Path("C:/Projects/Meaning-Oriented invariance Framework/moif/data/case")
OUT_DIR = "phase_m_r_results"
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
        'sub_id': sub_id,
        'EDA_Phasic': win(eda_phasic), 'HRV_RRI': win(rri_100),
        'Arousal': win(arousal), 'Valence': win(valence),
        'video_id': win(video).astype(int),
        'time_s': np.arange(nw)
    })
    scaler = StandardScaler()
    res_df[['EDA_Phasic', 'HRV_RRI']] = scaler.fit_transform(res_df[['EDA_Phasic', 'HRV_RRI']])
    return res_df[res_df['video_id'] != 0].copy()

# ── Entropy Audit ───────────────────────────────────────────────────────────
def calculate_entropies(full_df, k_val, shuffle=False):
    if shuffle:
        # Shuffle within subject to preserve marginals but break structure
        shuffled_df = full_df.copy()
        for sid in full_df['sub_id'].unique():
            idx = shuffled_df[shuffled_df['sub_id'] == sid].index
            shuffled_df.loc[idx, 'A_cat'] = np.random.permutation(shuffled_df.loc[idx, 'A_cat'])
            shuffled_df.loc[idx, 'V_cat'] = np.random.permutation(shuffled_df.loc[idx, 'V_cat'])
        df_eval = shuffled_df
    else:
        df_eval = full_df

    feats = full_df[['EDA_Phasic', 'HRV_RRI']].values
    nn = NearestNeighbors(n_neighbors=k_val + 100, metric='euclidean')
    nn.fit(feats)
    distances, indices = nn.kneighbors(feats)
    
    h_a, h_v = [], []
    for i in range(len(full_df)):
        target_sub = full_df.iloc[i]['sub_id']
        target_time = full_df.iloc[i]['time_s']
        neighbor_indices = indices[i]
        
        valid_indices = []
        for ni in neighbor_indices:
            if ni == i: continue
            n_row = full_df.iloc[ni]
            if n_row['sub_id'] == target_sub and abs(n_row['time_s'] - target_time) < 30:
                continue
            valid_indices.append(ni)
            if len(valid_indices) >= k_val: break
            
        if len(valid_indices) < k_val: continue
        
        n_df = df_eval.iloc[valid_indices]
        def ent(arr): return entropy(arr.value_counts(normalize=True), base=2)
        h_a.append(ent(n_df['A_cat']))
        h_v.append(ent(n_df['V_cat']))
        
    return np.mean(h_a), np.mean(h_v)

def run_red_team_audit():
    all_data = []
    for i in range(1, 16): # Process 15 subjects for M1-R for speed
        sid = f"sub_{i}"
        df = get_subject_data(sid)
        if df is not None: all_data.append(df)
    full_df = pd.concat(all_data).reset_index(drop=True)
    
    def discretize(v): return np.digitize(v, bins=[-0.33, 0.33])
    full_df['A_cat'] = discretize(full_df['Arousal'].values)
    full_df['V_cat'] = discretize(full_df['Valence'].values)
    
    # Global Entropies
    def global_ent(arr): return entropy(arr.value_counts(normalize=True), base=2)
    h_a_global = global_ent(full_df['A_cat'])
    h_v_global = global_ent(full_df['V_cat'])
    
    audit_results = {}
    for k in [20, 50, 100]:
        print(f"Auditing k={k}...")
        ha_true, hv_true = calculate_entropies(full_df, k, shuffle=False)
        ha_shuf, hv_shuf = calculate_entropies(full_df, k, shuffle=True)
        
        audit_results[f"k_{k}"] = {
            "H_A_true": ha_true, "H_V_true": hv_true,
            "H_A_shuf": ha_shuf, "H_V_shuf": hv_shuf,
            "H_A_norm": ha_true / h_a_global,
            "H_V_norm": hv_true / h_v_global,
            "delta_true": hv_true - ha_true,
            "delta_norm": (hv_true / h_v_global) - (ha_true / h_a_global)
        }
        
    final_report = {
        "global_entropy": {"Arousal": h_a_global, "Valence": h_v_global},
        "audit": audit_results,
        "gates": {
            "R1_normalized_gap": all(v['delta_norm'] > 0 for v in audit_results.values()),
            "R2_structure_check": all(v['H_V_true'] < v['H_V_shuf'] for v in audit_results.values()),
            "R3_k_stability": all(v['delta_true'] > 0 for v in audit_results.values())
        }
    }
    
    with open(os.path.join(OUT_DIR, "phase_m1r_audit_report.json"), "w") as f:
        json.dump(final_report, f, indent=4)
    print("Phase M1-R Audit Complete.")

if __name__ == "__main__":
    run_red_team_audit()
