import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.interpolate import interp1d

CASE_DIR = Path("C:/Projects/Meaning-Oriented invariance Framework/moif/data/case")
OUT_DIR = "phase_m_results"
os.makedirs(OUT_DIR, exist_ok=True)

def get_subject_counts(sub_id):
    a_path = CASE_DIR / 'data' / 'interpolated' / 'annotations' / f"{sub_id}.csv"
    if not a_path.exists(): return None
    df_a = pd.read_csv(a_path)
    
    def rescale(v): return (v - 5.0) / 4.0
    ws = 20
    nw = len(df_a) // ws
    def win(a): return a[:nw*ws].reshape(nw, ws).mean(axis=1)
    
    arousal = win(rescale(df_a['arousal'].values))
    valence = win(rescale(df_a['valence'].values))
    video = win(df_a['video'].values).astype(int)
    
    df = pd.DataFrame({'A': arousal, 'V': valence, 'vid': video})
    df_v = df[df['vid'] != 0]
    if len(df_v) == 0: return None
    
    # Thresholds (Subject-specific)
    a_hi = np.percentile(df_v['A'], 70)
    v_hi = np.percentile(df_v['V'], 70)
    v_lo = np.percentile(df_v['V'], 30)
    
    return {
        'sub_id': sub_id,
        'N_total': len(df_v),
        'N_HA': len(df_v[df_v['A'] > a_hi]),
        'N_HA_Pos': len(df_v[(df_v['A'] > a_hi) & (df_v['V'] > v_hi)]),
        'N_HA_Neg': len(df_v[(df_v['A'] > a_hi) & (df_v['V'] < v_lo)])
    }

def run_balance():
    all_res = []
    for i in range(1, 31):
        res = get_subject_counts(f"sub_{i}")
        if res: all_res.append(res)
    df = pd.DataFrame(all_res)
    df.to_csv(os.path.join(OUT_DIR, "phase_m2_label_balance_report.csv"), index=False)
    
    valid_subs = df[(df['N_HA_Pos'] >= 20) & (df['N_HA_Neg'] >= 20)]
    print(f"Phase M2 Label Balance Report: {len(valid_subs)}/30 subjects valid (N>=20 per class).")

if __name__ == "__main__":
    run_balance()
