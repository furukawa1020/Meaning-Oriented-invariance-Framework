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
if os.environ.get("ALLOW_REAL_DATA_PHASE_M1_AUDIT") != "1":
    print("FATAL: Phase M1 is locked.")
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
        'sub_id': sub_id,
        'EDA_Phasic': win(eda_phasic), 'HRV_RRI': win(rri_100),
        'Arousal': win(arousal), 'Valence': win(valence),
        'video_id': win(video).astype(int),
        'time_s': np.arange(nw)
    })
    
    # Z-score within subject
    scaler = StandardScaler()
    res_df[['EDA_Phasic', 'HRV_RRI']] = scaler.fit_transform(res_df[['EDA_Phasic', 'HRV_RRI']])
    
    # Exclude baseline
    return res_df[res_df['video_id'] != 0].copy()

# ── Neighborhood Analysis ───────────────────────────────────────────────────
def run_neighborhood_audit():
    all_data = []
    for i in range(1, 31):
        sid = f"sub_{i}"
        print(f"Loading CASE {sid}...")
        df = get_subject_data(sid)
        if df is not None: all_data.append(df)
    full_df = pd.concat(all_data).reset_index(drop=True)
    
    feats = full_df[['EDA_Phasic', 'HRV_RRI']].values
    
    # Discretize for Entropy (into 3 bins: -1 to -0.33, -0.33 to 0.33, 0.33 to 1.0)
    def discretize(v):
        return np.digitize(v, bins=[-0.33, 0.33])
    
    full_df['A_cat'] = discretize(full_df['Arousal'].values)
    full_df['V_cat'] = discretize(full_df['Valence'].values)
    
    # Fit kNN
    k_target = 50
    # Increase k slightly because we will filter out temporal neighbors
    nn = NearestNeighbors(n_neighbors=k_target + 100, metric='euclidean')
    nn.fit(feats)
    distances, indices = nn.kneighbors(feats)
    
    results = []
    for i in range(len(full_df)):
        target_sub = full_df.iloc[i]['sub_id']
        target_time = full_df.iloc[i]['time_s']
        
        neighbor_indices = indices[i]
        
        # Apply Temporal Exclusion (±30s if same subject)
        valid_neighbors = []
        for ni in neighbor_indices:
            if ni == i: continue # Self
            n_row = full_df.iloc[ni]
            if n_row['sub_id'] == target_sub:
                if abs(n_row['time_s'] - target_time) < 30:
                    continue # Temporal exclusion
            valid_neighbors.append(ni)
            if len(valid_neighbors) >= k_target:
                break
        
        if len(valid_neighbors) < k_target: continue
        
        # Calculate Local Entropy
        n_df = full_df.iloc[valid_neighbors]
        
        def calc_h(cat_arr):
            probs = cat_arr.value_counts(normalize=True)
            return entropy(probs, base=2)
            
        results.append({
            'H_Arousal': calc_h(n_df['A_cat']),
            'H_Valence': calc_h(n_df['V_cat'])
        })
        
    res_df = pd.DataFrame(results)
    summary = {
        "mean_H_Arousal": float(res_df['H_Arousal'].mean()),
        "mean_H_Valence": float(res_df['H_Valence'].mean()),
        "delta_H_V_A": float(res_df['H_Valence'].mean() - res_df['H_Arousal'].mean()),
        "gate_m1_pass": bool((res_df['H_Valence'].mean() - res_df['H_Arousal'].mean()) >= 0.05)
    }
    
    with open(os.path.join(OUT_DIR, "phase_m1_neighborhood_report.json"), "w") as f:
        json.dump(summary, f, indent=4)
    print("Phase M1 Neighborhood Report generated.")

if __name__ == "__main__":
    run_neighborhood_audit()
